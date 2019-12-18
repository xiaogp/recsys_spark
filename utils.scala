import java.io.FileInputStream
import java.security.MessageDigest
import java.time.format.DateTimeFormatter
import java.time.{LocalDate, ZoneId}
import java.util.Properties

import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.Put
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapred.TableOutputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, udf}


object utils {

  /**
   * 获取配置文件
   *
   * @param proPath
   * @return
   */
  def getProPerties(proPath: String) = {
    val properties: Properties = new Properties()
    properties.load(new FileInputStream(proPath))
    properties
  }


  /**
   * 求前多少天的日期
   *
   * @param dayDiff   时间窗口
   * @param startDate 起始时间
   * @return
   */
  def getDiffDatetime(dayDiff: Int, startDate: Any = None): String = {
    var date = LocalDate.now(ZoneId.systemDefault())
    if (startDate != None) {
      date = LocalDate.parse(startDate.toString, DateTimeFormatter.ofPattern("yyyy-MM-dd"))
    }
    date.plusDays(-dayDiff).toString()
  }

  /**
   * 计算recall和precison，阈值为0.5
   *
   * @param data
   * @param labelCol
   * @param predCol
   */
  def PREvaluation(data: DataFrame, labelCol: String, predCol: String): Unit = {
    val dataPred = data.select(col(predCol).cast("Double"), col(labelCol).cast("Double"))
      .rdd.map(row => (row.getDouble(0), row.getDouble(1)))
    val prMetrics = new MulticlassMetrics(dataPred)
    println(f"accuracy: ${prMetrics.accuracy}%.3f")
    println(f"precision: ${prMetrics.precision(1)}%.3f")
    println(f"recall: ${prMetrics.recall(1)}%.3f")
    println(f"fMeasure: ${prMetrics.fMeasure(1)}%.3f")
  }

  /**
   * 计算AUC
   *
   * @param data     预测数据
   * @param labelCol 实际标签
   * @param predCol  预测标签
   * @return
   */
  def AUCEvaluation(data: DataFrame, labelCol: String, predCol: String): Unit = {
    val metrics = new BinaryClassificationEvaluator()
      .setLabelCol(labelCol)
      .setRawPredictionCol(predCol)
      .setMetricName("areaUnderROC")
    val auc = metrics.evaluate(data)
    println(f"AreaUnderROC: ${auc}%.3f")
  }

  /**
   * 获得正类预测概率
   *
   * @param data   原始数据
   * @param preCol 预测列
   * @return
   */
  def getProba(data: DataFrame, preCol: String): DataFrame = {
    val probaFunc = udf((proba: Vector) => (proba(1)))
    data.withColumn("predProba", probaFunc(col(preCol)))
  }

  /**
   * spark dataframe保存到mysql
   *
   * @param data
   * @param host
   * @param port
   * @param database
   * @param user
   * @param password
   * @param saveMode
   * @param tableName
   */
  def dfSavePGSQL(data: DataFrame, host: String, port: String = "3306", database: String,
                  user: String, password: String, saveMode: String = "Append", tableName: String): Unit = {
    val properties: Properties = new Properties()
    properties.setProperty("password", password)
    properties.setProperty("user", user)
    properties.setProperty("url", s"jdbc:postgresql://${host}:${port}/${database}??useUnicode=true&characterEncoding=utf8&autoReconnect=true&useSSL=false")
    data.write.mode(saveMode).jdbc(properties.getProperty("url"), tableName, properties)
  }

  /**
   * md5加密
   *
   * @param content
   * @param sub_length
   * @return
   */
  def hashMD5(content: String, sub_length: Any = None): String = {
    val md5 = MessageDigest.getInstance("MD5")
    var encoded = md5.digest((content).getBytes).map("%02x".format(_)).mkString
    if (sub_length != None) {
      encoded = encoded.substring(0, sub_length.toString.toInt) + "_" + content
    }
    encoded
  }


  /**
   * key=>Array[String]保存到hbase
   *
   * @param data
   * @param hbaseTableName
   */
  def saveHbase(data: DataFrame, hbaseTableName: String, addSalt: Boolean = true): Unit = {
    val config = HBaseConfiguration.create()
    val jobConf = new JobConf(config)
    jobConf.setOutputFormat(classOf[TableOutputFormat])
    jobConf.set(TableOutputFormat.OUTPUT_TABLE, hbaseTableName)

    data.rdd.map(row => {
      val item_id = if (addSalt) hashMD5(row(0).toString, 5) else row(0).toString
      val similar_items_array = row.getAs[scala.collection.mutable.WrappedArray[String]](1).mkString("[", ",", "]")
      val put = new Put(Bytes.toBytes(item_id))
      for (i <- Array(similar_items_array).zip(Array("recommend"))) {
        put.addColumn(Bytes.toBytes("info"), Bytes.toBytes(i._2), Bytes.toBytes(i._1))
      }
      (new ImmutableBytesWritable, put)
    }).saveAsHadoopDataset(jobConf)
  }

}





