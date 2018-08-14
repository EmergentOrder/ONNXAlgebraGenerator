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

  @SuppressWarnings(Array("org.wartremover.warts.Equals"))
  implicit final class AnyOps[A](self: A) {
    def ===(other: A): Boolean = self == other
  }

  val path = Paths.get("src/main/scala/ONNXAlgebra.scala");

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
    val typeStringMap = x._2.map{b =>
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
                  case s if s.startsWith("tensor(") =>  {
                    val a = s.stripPrefix("tensor(").dropRight(1)
                      (typeConstraintParam.type_param_str.getString + "Tensor" + a.capitalize -> (typeConstraintParam.type_param_str.getString, "  type " + typeConstraintParam.type_param_str.getString + "Tensor" + a.capitalize + " = Tensor[" + a.capitalize.replaceAll("Int32", "Int")
            .replaceAll("Int64", "Long")
            .replaceAll("Int16", "Short")
            .replaceAll("Int8", "Byte")
            .replaceAll("Uint64", "ULong")
            .replaceAll("Uint32", "UInt")
            .replaceAll("Uint16", "UShort")
            .replaceAll("Uint8", "UByte")
            .replaceAll("Bool", "Boolean") + "]" + "\n"))
                  }

        case s => {
            (typeConstraintParam.type_param_str.getString + s.capitalize -> (typeConstraintParam.type_param_str.getString, "  type " + typeConstraintParam.type_param_str.getString + s.capitalize + " =" + s.capitalize.replaceAll("Int32", "Int")
            .replaceAll("Int64", "Long")
            .replaceAll("Int16", "Short")
            .replaceAll("Int8", "Byte")
            .replaceAll("Uint64", "ULong")
            .replaceAll("Uint32", "UInt")
            .replaceAll("Uint16", "UShort")
            .replaceAll("Uint8", "UByte")
            .replaceAll("Bool", "Boolean") + "\n"))
                }
                }


                )

        allowedTypeStrings
      }
    }.flatten
      .flatten
      .toMap
   
//      typeStringMap
//  }

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
          .replaceAll("Tensor", "Tensor[VV]") + ")" + (if (required)
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

    val maxSinceVersion = (x._2.map(z => z._2) foldLeft 0)(Math.max)

        val beginString = "@free trait " + x._1 + " extends Operator" + " {\n"

      //  val opts = optionalInputs.map(g => g.map(h => h.GetTypeStr.getString))
      //  if(x._1 === "Add") println(opts)
        val defStrings = (0 until 
          requiredInputs.size).map {z =>
   //       scala.math.max(requiredInputs.map(g => g.map(h => h.GetTypeStr.getString)).distinct.size,
     //       attributesStrings.distinct.size)).map{z =>
                             //                    optionalInputs.map(g => g.map(h => h.GetTypeStr.getString)).distinct.size)).map {z =>
      "\n  def " + x._1 +
//      (if(x._2(z)._2 < maxSinceVersion) x._2(z)._2.toString else "") +
      x._2(z)._2.toString +
      "[VV : spire.math.Numeric:ClassTag](" + 
      "name: String" +
      (if (requiredInputs(z).size > 0 || optionalInputs(z).size > 0) "," else "") +
      requiredInputs(z)
        .map(y =>
          y.GetName.getString
            .replaceAll("var", "someVar") + ": " + "" + y.GetTypeStr.getString
            .replaceAll("B", "T").replaceAll("V", "T").replaceAll("I", "T")
            .replaceAll("T1", "T").replaceAll("Tind", "T").replaceAll("T", "T[VV]").replaceAll("tensor\\(int64\\)", "Tensor[Long]") + ", " + y.GetName.getString + "name: String"
            )
        .mkString(", ") +
      (if (requiredInputs(z).size > 0 && optionalInputs(z).size > 0) "," else "") +
      optionalInputs(z)
        .map(y =>
          y.GetName.getString
            .replaceAll("var", "someVar")
            .replaceAll("shape", "shapeInput") + ": " + "Option[" + y.GetTypeStr.getString
            .replaceAll("T1", "T").replaceAll("Tind", "T").replaceAll("T", "T[VV]").replaceAll("tensor\\(int64\\)", "Tensor[Long]") + "] = None")
        .mkString(", ") +
      (if (attributesStrings(z).size > 0 && (requiredInputs(z).size + optionalInputs(z).size) > 0)
         "," + attributesStrings(z)
       else "") +
      ")\n" + "    : FS[(" + (0 until x._2(z)._4.size.toInt)
      .map(y => x._2(z)._4.get(y))
      .map(y =>
        "" + y.GetTypeStr.getString.replaceAll("T2", "T")
           .replaceAll("B", "T").replaceAll("V", "T").replaceAll("I", "T")
           .replaceAll("T1", "T").replaceAll("T", "T[VV]").replaceAll("tensor\\(int64\\)",
                                                       "Tensor[Long]"))
      .mkString(", ") + ")]\n"
      }.distinct.mkString("\n") 
      val endString = "\n}"

      val traitString = beginString + defStrings + endString

    (traitString, typeStringMap)
  }

  val flattenedTypeStringsMap =
    traitStringsAndTypeStrings.map(x => x._2).flatten.toMap
  val typeStrings = flattenedTypeStringsMap.values
    .map(z => z._2)
    .mkString("\n") + flattenedTypeStringsMap.values
    .map(z => "  type " + z._1 + "[VV] = Tensor[VV]")
    .toList
    .distinct
    .mkString("\n")
  val traitStrings = traitStringsAndTypeStrings
    .map(x => x._1)
    .filter(x => !x.contains("ATen"))
    .mkString("\n")

  val fullSource = "package org.emergentorder\n\n" +
    "import freestyle.free._\n" +
    "import freestyle.free.implicits._\n" +
//    "import spire.math.Number\n" +
    "import spire.math.UByte\n" +
    "import spire.math.UShort\n" +
    "import spire.math.UInt\n" +
    "import spire.math.ULong\n" +
    "import spire.math._\n" +
    "import spire.implicits._\n" +
    "import scala.reflect.ClassTag\n" +
    "import scala.language.higherKinds\n\n" +
    "package object onnx {\n" +
    "  type Tensor[U] = Tuple2[Vector[U], Seq[Int]]\n" +
    "  trait Operator\n" +
    typeStrings + "\n" +
//    "}\n" +
    "@free trait DataSource {\n" +
    "  def inputData[VV:spire.math.Numeric:ClassTag]: FS[Tensor[VV]]\n" +
    "  def getParams[VV:spire.math.Numeric:ClassTag](name: String): FS[Tensor[VV]]\n" +
    "  def getAttributes[VV:spire.math.Numeric:ClassTag](name: String): FS[Tensor[VV]]\n" +
    "}\n" +
    traitStrings +
    "}\n"

  def generate(): Unit = {
    val onnxSource = fullSource.parse[Source].get
    val wrote = Files.write(path, onnxSource.syntax.getBytes("UTF-8"));
  }

  generate()
}
