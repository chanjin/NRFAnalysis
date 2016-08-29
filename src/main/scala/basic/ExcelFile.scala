package basic

/**
  * Created by chanjinpark on 2016. 6. 16..
  */

import java.io._

import org.apache.poi.ss.usermodel.Cell
import org.apache.poi.xssf.usermodel.{XSSFCell, XSSFRow, XSSFWorkbook}

object ExcelFile {
  val workspace ="/Users/chanjinpark/data/NRF2015/"
  val file = workspace + "NRF2015.xlsx"
  val colId = 21
  val colName = 22
  val colContStart = 65
  val colContEnd = 69
  val sep =","

  val metadata = workspace + "NRF2015Meta.csv"
  val contdir = workspace + "content/"

  def removeCommas(s: String) =
    s.replaceAll(",", " ").replaceAll("[\n”\"]", "")


  private def matchCell(cell: XSSFCell) = {
    cell.getCellType match {
      case Cell.CELL_TYPE_STRING => cell.getStringCellValue
      case Cell.CELL_TYPE_NUMERIC => cell.getRawValue
      case Cell.CELL_TYPE_BLANK => ""
      case _ => "*******"
    }
  }

  def writeRow(row: XSSFRow, ncols: Int) : String = {
    val name = row.getCell(21).getStringCellValue
    val content = (colContStart until colContEnd + 1).map(j => matchCell(row.getCell(j))).mkString("\n\n")

    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(contdir + name + ".txt"), "UTF-8"));
    writer.write(content)
    writer.close()

    (0 until colContStart).map(j => matchCell(row.getCell(j))).mkString(sep)
  }

  def writeRowOnlyMeta(row: XSSFRow, ncols: Int) : String = {
    val name = row.getCell(21).getStringCellValue
    (0 until colContStart).map(j => matchCell(row.getCell(j))).mkString(sep)
  }

  def main(args: Array[String]) = {
    import java.io.FileInputStream
    val wb = new XSSFWorkbook(new FileInputStream(file))
    val sheet = wb.getSheetAt(0)
    val header = sheet.getHeader
    val rowsCount = sheet.getLastRowNum

    println("Total Number of Rows: " + (rowsCount + 1))
    //(2 until rowsCount + 1).foreach(i => {

    val writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(metadata), "UTF-8"));
    //new PrintWriter(new File(metadata))
    (2 until rowsCount + 1).foreach(i => {
      val row = sheet.getRow(i)
      val ncols = row.getLastCellNum
      if (ncols != 70) println("^^^^^" + row.getCell(13))
      writer.write(writeRowOnlyMeta(row, ncols) + "\n")
    })
    writer.close()

    println("Last - " + sheet.getRow(rowsCount).getCell(13))
  }

}
