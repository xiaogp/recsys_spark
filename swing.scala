import utils.{getDiffDatetime, getProPerties, saveHbase}

import scala.collection.mutable.ArrayBuffer
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.log4j.{Level, Logger}


object swing {
  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)
    val spark = SparkSession
      .builder
      .master("yarn")
      .appName("swing")
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._

    /**
     * window_days:  时间窗口
     * similar_item_num:  商品的候选相似商品数量
     * hot_item_regular:  热门商品惩罚力度
     * profile_decay:  用户偏好时间衰减率
     * black_user:  黑名单用户
     * black_items:  黑名单商品
     * recommend_num:  推荐商品数量
     * recommendSaveTable:  结果保存表
     */

    val properties = getProPerties(args(0))
    val window_days = properties.getProperty("window_days").toInt
    val similar_item_num = properties.getProperty("similar_item_num").toInt
    val hot_item_regular = properties.getProperty("hot_item_regular").toDouble
    val profile_decay = properties.getProperty("profile_decay").toDouble
    val black_users = properties.getProperty("black_users")
    val black_items = properties.getProperty("black_items")
    val recommend_num = properties.getProperty("recommend_num").toInt
    val start_date = getDiffDatetime(window_days)
    val table_date = getDiffDatetime(0)

    println(s"训练时间窗口:${start_date} => ${table_date}")

    val df_sales = spark.sql(s"select cast(USR_NUM_ID as bigint), cast(ITEM_NUM_ID as bigint), ORDER_DATE from gp_test.sales_data " +
      s"where to_date(ORDER_DATE) >= '${start_date}' and USR_NUM_ID not in (${black_users}) and ITEM_NUM_ID not in (${black_items})")
      .toDF("userid", "itemid", "date").cache()

    println(s"交易数量:${df_sales.count()}")

    // 构建用户购买序列
    val df_sales1 = df_sales.groupBy("userid").agg(collect_set("itemid").as("itemid_set"))

    // 商品共现矩阵，此处flatMap容易导致数据倾斜
    val df_sales2 = df_sales1.flatMap { row =>
      val itemlist = row.getAs[scala.collection.mutable.WrappedArray[Long]](1).toArray.sorted
      val result = new ArrayBuffer[(Long, Long, Long)]()
      for (i <- 0 to itemlist.length - 2) {
        for (j <- i + 1 to itemlist.length - 1) {
          result += ((itemlist(i), itemlist(j), 1)) // 热门user惩罚
        }
      }
      result // 将result展开,每一个元素一行
    }.withColumnRenamed("_1", "itemidI").withColumnRenamed("_2", "itemidJ").withColumnRenamed("_3", "score")

    // 商品组合至少被两个人买过的,并且商品的组合已经排过序了
    val df_sales3 = df_sales2.groupBy("itemidI", "itemidJ").agg(sum("score").as("score")).filter($"score" >= 2)

    // 商品的倒排表
    val df_item1 = df_sales.groupBy("itemid").agg(collect_set("userid").as("userid_set"))

    // 把商品的购买用户集合join进来，商品的组合也是排过序的
    val df_join1 = df_sales3.join(df_item1.withColumnRenamed("itemid", "itemidI").withColumnRenamed("userid_set", "userid_set_I"), Seq("itemidI"), "left_outer")
    val df_join2 = df_join1.join(df_item1.withColumnRenamed("itemid", "itemidJ").withColumnRenamed("userid_set", "userid_set_J"), Seq("itemidJ"), "left_outer")

    val df_join4 = df_join2.flatMap { row =>
      val itemidJ = row.getAs[Long]("itemidJ")
      val itemidI = row.getAs[Long]("itemidI")
      val score = row.getAs[Long]("score")
      val userid_set_I = row.getAs[scala.collection.mutable.WrappedArray[Long]]("userid_set_I").toArray
      val userid_set_J = row.getAs[scala.collection.mutable.WrappedArray[Long]]("userid_set_J").toArray
      // 两个用户集合取交集
      val pair_buy = userid_set_I.intersect(userid_set_J).sorted // 取交集
      val result = new ArrayBuffer[(Long, Long, Long, Long, Long)]()
      for (i <- 0 to pair_buy.length - 2) {
        for (j <- i + 1 to pair_buy.length - 1) {
          result += ((itemidI, itemidJ, score, pair_buy(i), pair_buy(j))) // 热门user惩罚
        }
      }
      result
    }.toDF("itemidI", "itemidJ", "score", "useridI", "useridJ")

    // 计算公式
    val df_item2 = df_item1.flatMap { row =>
      val userlist = row.getAs[scala.collection.mutable.WrappedArray[Long]](1).toArray.sorted
      val result = new ArrayBuffer[(Long, Long, Long)]()
      for (i <- 0 to userlist.length - 2) {
        for (j <- i + 1 to userlist.length - 1) {
          result += ((userlist(i), userlist(j), 1)) // 热门user惩罚
        }
      }
      result // 将result展开,每一个元素一行
    }.withColumnRenamed("_1", "useridI").withColumnRenamed("_2", "useridJ").withColumnRenamed("_3", "user_pair_score")

    // 共同购买的商品数量
    val df_item3 = df_item2.groupBy("useridI", "useridJ").agg(sum("user_pair_score").as("user_pair_score"))

    // join
    val df_join5 = df_join4.join(df_item3, Seq("useridI", "useridJ"), "left_outer")

    // 最后一步计算,分母平滑因子取了1
    val df_join6 = df_join5.withColumn("similar", lit(1) / (lit(1) + $"user_pair_score")).select("itemidI", "itemidJ", "similar").withColumn("rank", row_number().over(Window.partitionBy("itemidI").orderBy($"similar".desc))).filter(s"rank <= ${similar_item_num}").drop("rank")

    // 再union一下
    val df_join7 = df_join6.union(df_join6.select($"itemidJ".as("itemidI"), $"itemidI".as("itemidJ"), $"similar"))
    val df_join8 = df_join7.groupBy("itemidI", "itemidJ").agg(bround(sum("similar"), 5).as("similar"))
    val swing_similar = df_join8.map { row =>
      val itemidI = row.getLong(0)
      val itemidJ_similar = (row.getLong(1).toString, row.getDouble(2))
      (itemidI, itemidJ_similar)
    }.toDF("itemid", "similar_items").groupBy("itemid").agg(collect_list("similar_items").as("similar_items"))

    // 用户的偏好
    val score = df_sales.withColumn("pref", lit(1) / (datediff(current_date(), $"date") * profile_decay + 1)).groupBy("userid", "itemid").agg(sum("pref").as("pref"))

    // 内连接，会连接所有可能
    val df_user_prefer1 = df_join8.join(score, $"itemidI" === $"itemid", "inner")

    // 偏好 × 相似度 × 商品热度降权
    val df_user_prefer2 = df_user_prefer1.withColumn("score", col("pref") * col("similar")).select("userid", "itemidJ", "score")

    // 取推荐top，把已经购买过的去除
    val df_user_prefer3 = df_user_prefer2.groupBy("userid", "itemidJ").agg(sum("score").as("score")).withColumnRenamed("itemidJ", "itemid")
    val df_user_prefer4 = df_user_prefer3.join(score, Seq("userid", "itemid"), "left").filter("pref is null")
    val swing_recommend = df_user_prefer4.select("userid", "itemid", "score").withColumn("rank", row_number().over(Window.partitionBy("userid").orderBy($"score".desc))).filter(s"rank <= ${recommend_num}").groupBy("userid").agg(collect_list("itemid").as("recommend"))

    saveHbase(swing_similar, "SWING_SIMILAR")
    saveHbase(swing_recommend, "SWING_RECOMMEND")

  }
}




