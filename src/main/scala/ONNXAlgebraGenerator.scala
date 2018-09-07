/*
 * ONNXAlgebraGenerator
 * Copyright (c) 2018 Alexander Merritt
 * All rights reserved. 
 * This program is free software: you can redistribute it and/or modify
 *
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

package org.emergentorder.onnx

import scala.meta._
import java.nio.file.Files
import java.nio.file.Paths

object ONNXAlgebraGenerator extends App {

  val useFS = true

  @SuppressWarnings(Array("org.wartremover.warts.Equals"))
  implicit final class AnyOps[A](self: A) {
    def ===(other: A): Boolean = self == other
  }

  val path = Paths.get("src/main/scala/ONNXAlgebra" + (if(useFS) "FS" else "") + ".scala");

  val attrMap = Array("Float",
                      "Int",
                      "String",
                      "Tensor",
                      "GRAPH???",
                      "Seq[Float]",
                      "Seq[Int]",
                      "Seq[String]",
                      "Seq[Tensor]",
                      "Seq[GRAPH???]").zipWithIndex.toMap
  val attrTypeMap = for ((k, v) <- attrMap) yield (v, k)

  val loaded =
    org.bytedeco.javacpp.Loader.load(classOf[org.bytedeco.javacpp.onnx])

  val schemas = org.bytedeco.javacpp.onnx.OpSchemaRegistry.get_all_schemas_with_history
  val schemasSize = schemas.size
  val scalaCollSchemas = (0 until schemasSize.toInt).map(x => schemas.get(x))
  val tuples = scalaCollSchemas.map(
    x =>
      (x.Name.getString,
       x.since_version,
       x.inputs,
       x.outputs,
       x.typeConstraintParams,
       x.attributes)).groupBy(_._1)
         //.map(y => (y._1, y._2.distinct))
//  tuples.foreach( y => println(y._2))

  val traitStringsAndTypeStrings = tuples.map { x =>
    val typeStringMap: Map[String, IndexedSeq[String]] = x._2.map{b =>
      val typeConstraintParams = b._5
      (0 until typeConstraintParams.size.toInt)
      .map { y =>
        val typeConstraintParam = typeConstraintParams.get(y)
        val allowedTypeStrings =
          (0 until typeConstraintParam.allowed_type_strs.size.toInt).map(
            z =>
              typeConstraintParam.allowed_type_strs
                .get(z)
                .getString match
                {
                  //TODO: collapse cases
                  case s if s.startsWith("tensor(") =>  {
                    val a = s
                      (typeConstraintParam.type_param_str.getString, a.capitalize
            .replaceAll("uint64", "ULong")
            .replaceAll("uint32", "UInt")
            .replaceAll("uint16", "UShort")
            .replaceAll("uint8", "UByte")
            .replaceAll("int64", "Long")
            .replaceAll("Int64", "Long")
            .replaceAll("int32", "Int")
            .replaceAll("Int32", "Int")
            .replaceAll("int16", "Short")
            .replaceAll("int8", "Byte")
            .replaceAll("string", "String")
            .replaceAll("float", "Float")
            .replaceAll("double", "Double")
            .replaceAll("Bool", "Boolean")
            .replaceAll("bool", "Boolean")
            .replaceAll("complex64", "Complex[Float]")
            .replaceAll("complex128", "Complex[Double]")
            )
                  }

        case s => {
            (typeConstraintParam.type_param_str.getString, s.capitalize.replaceAll("\\(", "[").replaceAll("\\)", "]").replaceAll("map", "Map")
            .replaceAll("uint64", "ULong")
            .replaceAll("uint32", "UInt")
            .replaceAll("uint16", "UShort")
            .replaceAll("uint8", "UByte")
            .replaceAll("int32", "Int")
            .replaceAll("Int32", "Int")
            .replaceAll("int64", "Long")
            .replaceAll("Int64", "Long")
            .replaceAll("int16", "Short")
            .replaceAll("int8", "Byte")
            .replaceAll("string", "String")
            .replaceAll("float", "Float")
            .replaceAll("double", "Double")
            .replaceAll("Bool", "Boolean")
            .replaceAll("bool", "Boolean")
            .replaceAll("complex64", "Complex[Float]")
            .replaceAll("complex128", "Complex[Double]")
            )
                }
                }


                )
        println(allowedTypeStrings)
        allowedTypeStrings
      }
    }
      .flatten
      .distinct
      .flatten
      .groupBy(_._1)
      .map{ case(key, value) => (key, value.map(_._2))}
      //      .flatten
        .toMap //+ ("tensor(int64)" -> IndexedSeq("Tensor[Long]")) + ("tensor(float)" -> IndexedSeq("Tensor[Float]"))

   
//      typeStringMap
//  }

println(typeStringMap)
    //CAUTION: iterator, unsafe
//    val attrIter = x._6.begin
    val attributesStrings: Seq[String] = x._2.map{z =>
        val attrIter = z._6.begin
        (0 until z._6.size.toInt)
      .map { y =>
        val result =
          (attrIter.first.getString, attrTypeMap(attrIter.second.`type`))
        val required = attrIter.second.required
        val incremented = attrIter.increment
        val str = "" + result._1
          .replaceAll("split", "splitAttr")
          .replaceAll("scale", "scaleAttr") + " : " + (if (required) ""
                                                       else
                                                         "Option[") + "(" + result._2
          .replaceAll("Tensor", "Tensor[_]") + 
          ")" + 
          (if (required)
                                                                     ""
                                                                   else
                                                                     "] = None")
        str
      }
      .mkString(",")
    }

    val requiredInputs = x._2.map{y =>
      (0 until y._3.size.toInt)
      .map(z => y._3.get(z))
      .filter(z => z.GetOption === 0)
    }
    val optionalInputs = x._2.map{y =>
      (0 until y._3.size.toInt)
      .map(z => y._3.get(z))
      .filter(z => z.GetOption === 1)
    }
  
    val outputs = x._2.map{y =>
      (0 until y._4.size.toInt)
      .map(z => y._4.get(z))
    }

    val maxSinceVersion = (x._2.map(z => z._2) foldLeft 0)(Math.max)

        val beginString = (if(useFS) "@free " else "") + "trait " + x._1 + 
          (if(useFS) "FS" else "") + " extends Operator" + (if(useFS) " with " + x._1 else "") + " {\n"

      //  val opts = optionalInputs.map(g => g.map(h => h.GetTypeStr.getString))
      //  if(x._1 === "Add") println(opts)
        val defStrings = (0 until 
          requiredInputs.size).map {z =>
   //       scala.math.max(requiredInputs.map(g => g.map(h => h.GetTypeStr.getString)).distinct.size,
     //       attributesStrings.distinct.size)).map{z =>
                             //                    optionalInputs.map(g => g.map(h => h.GetTypeStr.getString)).distinct.size)).map {z =>
      "\n  def " + x._1 + x._2(z)._2.toString +
//      (if(x._2(z)._2 < maxSinceVersion) x._2(z)._2.toString else "") +
/*      x._2(z)._2.toString + (if (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0) "[" else "") +
        (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
        .map(y =>
        y.GetTypeStr.getString + " : Numeric:ClassTag:" + "(" 
                       + typeStringMap(y.GetTypeStr.getString).map(_.replaceAll("\\(", "[").replaceAll("\\)", "]"))
                          .mkString(" |: ")
                       + ")"  //+ "#check"
                       ) ++
        optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
          .map(y =>
              y.GetTypeStr.getString + " : Numeric:ClassTag:" + "(" + typeStringMap(y.GetTypeStr.getString).map(_.replaceAll("\\(", "[").replaceAll("\\)", "]"))
                        .mkString(" |: ")  + ")" //+ "#check" 
                        ) ++
        outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
        .map(y =>
        y.GetTypeStr.getString + " : Numeric:ClassTag:" + "(" 
                       + typeStringMap(y.GetTypeStr.getString).map(_.replaceAll("\\(", "[").replaceAll("\\)", "]"))
                          .mkString(" |: ")
                       + ")"  //+ "#check"
                       )).distinct.mkString(",") +
      (if (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0) "]" else "") + */
      "(" + 
      "name: String" +
      (if (requiredInputs(z).size > 0 || optionalInputs(z).size > 0) "," else "") +
      requiredInputs(z)
        .map(y =>
            y.GetName.getString.replaceAll("var", "someVar") 
                       + ": " + (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString)) typeStringMap(y.GetTypeStr.getString).map(_.replaceAll("\\(", "[").replaceAll("\\)", "]")).mkString(" |: ")
                         else y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long]").replaceAll("tensor\\(float\\)","Tensor[Float]"))
            + ", " + y.GetName.getString + "name: String"
            )
        .mkString(", ") +
      (if (requiredInputs(z).size > 0 && optionalInputs(z).size > 0) "," else "") +
      optionalInputs(z)
        .map(y =>
          y.GetName.getString
            .replaceAll("var", "someVar")
            .replaceAll("shape", "shapeInput") 
            + ": " + "Option[" + (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString)) typeStringMap(y.GetTypeStr.getString).map(_.replaceAll("\\(", "[").replaceAll("\\)", "]")).mkString(" |: ")
                         else y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long]").replaceAll("tensor\\(float\\)","Tensor[Float]")) +
                         "] = None"
            )
        .mkString(", ") +
      (if (attributesStrings(z).size > 0 && (requiredInputs(z).size + optionalInputs(z).size) > 0)
         "," + attributesStrings(z)
       else "") +
      ")\n" + "    : " + //TODO: invoke def on parent trait
      (if(useFS) "FS[" else "") + "(" + 
      outputs(z)
      .map(y =>
        "" + (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString)) typeStringMap(y.GetTypeStr.getString).map(_.replaceAll("\\(", "[").replaceAll("\\)", "]")).mkString(" |: ")
                         else y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long]").replaceAll("tensor\\(float\\)","Tensor[Float]")) 
 )
      .mkString(", ") + ")" + (if(useFS) "]" else "") +"\n"
      }.distinct.mkString("\n") 
      val endString = "\n}"

      val traitString = beginString + defStrings + endString

    (traitString, typeStringMap)
  }

  val flattenedTypeStringsMap =
    traitStringsAndTypeStrings.map(x => x._2).flatten.toMap
//  val typeStrings = flattenedTypeStringsMap.values
//    .map(z => z._2)
//    .mkString("\n") + flattenedTypeStringsMap.values
//    .map(z => "  type " + z._1.replaceAll("T1", "t1").replaceAll("T2", "t2").replaceAll("Tind", "tind") + "[VV] = Tensor[VV]")
//    .toList
//    .distinct
//    .mkString("\n")

//  println(typeStrings)
  val traitStrings = traitStringsAndTypeStrings
    .map(x => x._1)
    .filter(x => !x.contains("ATen"))
    .mkString("\n")

  val fullSource = "package org.emergentorder\n\n" +
    (if(useFS) "import freestyle.free._\n" else "") +
    (if(useFS) "import freestyle.free.implicits._\n" else "") +
//    "import spire.math.Number\n" +
    "import spire.math.UByte\n" +
    "import spire.math.UShort\n" +
    "import spire.math.UInt\n" +
    "import spire.math.ULong\n" +
    "import spire.math.Complex\n" +
    "import spire.math.Numeric\n" +
//    "import spire.implicits._\n" +
    "import scala.reflect.ClassTag\n\n" +
//    "import scala.language.higherKinds\n\n" +
    "package" + (if(useFS) "" else " object") + " onnx " +
    "{\n" +
 (if(useFS) "" else "type |:[+A1, +A2] = Either[A1, A2]\n") +
    (if(useFS) "" else "  type Tensor[U] = Tuple2[Vector[U], Seq[Int]]\n") +
    (if(useFS) "" else "  trait Operator\n") +
//    (if(useFS) "" else typeStrings) + "\n" +
//    "}\n" +
    (if(useFS) "@free " else "")  +
    "trait DataSource" + (if(useFS) "FS extends DataSource" else "") + " {\n" +
    "  def inputData[VV:Numeric:ClassTag" + "]: " +
    (if(useFS) "FS[" else "") +
    "Tensor[VV]" + (if(useFS) "]" else "") +"\n" +
    "  def getParams[VV:Numeric:ClassTag" + "](name: String): " +
    (if(useFS) "FS[" else "") +
    "Tensor[VV]" + (if(useFS) "]" else "") +"\n" +
    "  def getAttributes[VV:Numeric:ClassTag"  + "](name: String): " +
    (if(useFS) "FS[" else "") +
    "Tensor[VV]" + (if(useFS) "]" else "") +"\n" +
    "}\n" +
    traitStrings +
    "}\n"

  def generate(): Unit = {
    println(fullSource)
    val onnxSource = fullSource.parse[Source].get
    val wrote = Files.write(path, onnxSource.syntax.getBytes("UTF-8"));
  }

  generate()
}
