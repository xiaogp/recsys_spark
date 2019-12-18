import utils.{getDiffDatetime, getProPerties, saveHbase}

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer


object itemCF {
  def main(args: Array[String]): Unit = {
    Logger.getRootLogger.setLevel(Level.WARN)
    val spark = SparkSession
      .builder
      .master("yarn")
      .appName("itemCF")
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._

    /**
     * window_days: 时间窗口
     * similar_item_num: 商品的候选相似商品数量
     * hot_item_regular: 热门商品惩罚力度
     * profile_decay: 用户偏好时间衰减率
     * black_user: 黑名单用户
     * black_items: 黑名单商品
     * recommend_num:  推荐商品数量
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

    val df_sales = spark.sql(s"select USR_NUM_ID, ITEM_NUM_ID, ORDER_DATE from gp_test.sales_data " +
      s"where to_date(ORDER_DATE) >= '${start_date}' and USR_NUM_ID not in (${black_users}) and ITEM_NUM_ID not in (${black_items})")
      .toDF("userid", "itemid", "date").cache()

    println(s"交易数量:${df_sales.count()}")

    // 构建用户购买序列
    val df_sales1 = df_sales.groupBy("userid").agg(collect_set("itemid").as("itemid_set"))

    // 商品共现矩阵
    val df_sales2 = df_sales1.flatMap { row =>
      val itemlist = row.getAs[scala.collection.mutable.WrappedArray[Int]](1).toArray.sorted
      val result = new ArrayBuffer[(Int, Int, Double)]()
      for (i <- 0 to itemlist.length - 2) {
        for (j <- i + 1 to itemlist.length - 1) {
          result += ((itemlist(i), itemlist(j), 1.0 / math.log(1 + itemlist.length))) // 热门user惩罚
        }
      }
      result
    }.withColumnRenamed("_1", "itemidI").withColumnRenamed("_2", "itemidJ").withColumnRenamed("_3", "score")

    val df_sales3 = df_sales2.groupBy("itemidI", "itemidJ").agg(sum("score").as("sumIJ"))

    // 计算商品的购买次数
    val df_sales0 = df_sales.withColumn("score", lit(1)).groupBy("itemid").agg(sum("score").as("score"))

    // 计算共现相似度,N ∩ M / srqt(N * M), row_number取top top_similar_item_num
    val df_sales4 = df_sales3.join(df_sales0.withColumnRenamed("itemid", "itemidJ").withColumnRenamed("score", "sumJ").select("itemidJ", "sumJ"), "itemidJ")
    val df_sales5 = df_sales4.join(df_sales0.withColumnRenamed("itemid", "itemidI").withColumnRenamed("score", "sumI").select("itemidI", "sumI"), "itemidI")
    val df_sales6 = df_sales5.withColumn("result", bround(col("sumIJ") / sqrt(col("sumI") * col("sumJ")), 5)).withColumn("rank", row_number().over(Window.partitionBy("itemidI").orderBy($"result".desc))).filter(s"rank <= ${similar_item_num}").drop("rank")

    // itme1和item2交换
    val df_sales8 = df_sales6.select("itemidI", "itemidJ", "sumJ", "result").union(df_sales6.select($"itemidJ".as("itemidI"), $"itemidI".as("itemidJ"), $"sumI".as("sumJ"), $"result")).withColumnRenamed("result", "similar").cache()
    val itemcf_similar = df_sales8.map { row =>
      val itemidI = row.getInt(0)
      val itemidJ_similar = (row.getInt(1).toString, row.getDouble(3))
      (itemidI, itemidJ_similar)
    }.toDF("itemid", "similar_items").groupBy("itemid").agg(collect_list("similar_items").as("similar_items"))

    // 计算用户偏好
    val score = df_sales.withColumn("pref", lit(1) / (datediff(current_date(), $"date") * profile_decay + 1)).groupBy("userid", "itemid").agg(sum("pref").as("pref"))

    // 内连接，会连接所有可能
    val df_user_prefer1 = df_sales8.join(score, $"itemidI" === $"itemid", "inner")

    // 偏好 × 相似度 × 商品热度降权
    val df_user_prefer2 = df_user_prefer1.withColumn("score", col("pref") * col("similar") * (lit(1) / log(col("sumJ") * hot_item_regular + math.E))).select("userid", "itemidJ", "score")

    // 取推荐top，把已经购买过的去除
    val df_user_prefer3 = df_user_prefer2.groupBy("userid", "itemidJ").agg(sum("score").as("score")).withColumnRenamed("itemidJ", "itemid")
    val df_user_prefer4 = df_user_prefer3.join(score, Seq("userid", "itemid"), "left").filter("pref is null")
    val itemcf_recommend = df_user_prefer4.select($"userid", $"itemid".cast("String"), $"score").withColumn("rank", row_number().over(Window.partitionBy("userid").orderBy($"score".desc))).filter(s"rank <= ${recommend_num}").groupBy("userid").agg(collect_list("itemid").as("recommend"))

    // 保存商品共现相似度数据
    saveHbase(itemcf_similar, "ITEMCF_SIMILAR")

    // 保存用户偏好推荐数据
    saveHbase(itemcf_recommend, "ITEMCF_RECOMMEND")

  }
}


