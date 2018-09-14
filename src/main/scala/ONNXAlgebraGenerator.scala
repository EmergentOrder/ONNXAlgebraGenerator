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

  //TODO: Fix op ordering in target source file
  val useFS = true
  val useDotty = false
  val unionTypeOperator = (if(useDotty) " | " else " TypeOr ")
  //Missing: Non-numeric, Boolean and String
  //TODO: The rest of the desugaring for the union type context bounds
  val checkedTypes ="(implicit ev:" + (if(useDotty) "(" else "(UNil TypeOr ") + "Float16" + unionTypeOperator + "Float" + unionTypeOperator + "Double" + unionTypeOperator + "Byte" + unionTypeOperator + "Short" + unionTypeOperator + "Int" + unionTypeOperator + "Long" + unionTypeOperator + "UByte" + unionTypeOperator + "UShort" + unionTypeOperator + "UInt" + unionTypeOperator + "ULong" + unionTypeOperator + "Complex[Float]" + unionTypeOperator + "Complex[Double]" + (if(useDotty) ")" else ")#check[T])")

  val inputTypes = "T " + (if(useDotty) "<: " else ": ")  + "Numeric:ClassTag:Field"

  @SuppressWarnings(Array("org.wartremover.warts.Equals"))
  implicit final class AnyOps[A](self: A) {
    def ===(other: A): Boolean = self == other
  }

  val path = Paths.get("src/main/scala/ONNXAlgebra" + (if(useFS) "Free" else "") + ".scala");

  //TODO: Fix fragility here, get it (indirectly) from the protobuf
  val attrMap = Array("Undefined",
                      "Float",
                      "Int",
                      "String",
                      "Tensor",
                      "Graph",
                      "Seq[Float]",
                      "Seq[Int]",
                      "Seq[String]",
                      "Seq[Tensor]",
                      "Seq[Graph]").zipWithIndex.toMap
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
          .replaceAll("Tensor", "Tensor[T, J]") + 
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
          (if(useFS) "Free" else "") + " extends Operator" + (if(useFS) " with " + x._1 else "") + " {\n"

      //  val opts = optionalInputs.map(g => g.map(h => h.GetTypeStr.getString))
      //  if(x._1 === "Add") println(opts)
        val defStrings = (0 until 
          requiredInputs.size).map {z =>
           val requiredImplicitsInputs = (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
           .map(y =>
           //TODO: extract
            "ev" + y.GetTypeStr.getString + ":" + (if(useDotty) "(" else "(UNil TypeOr ") + typeStringMap(y.GetTypeStr.getString).map{ a =>
              val replaceParens = a.replaceAll("\\(", "[").replaceAll("\\)", "]")
              (if(replaceParens.contains("Tensor[")) replaceParens.stripPrefix("Tensor[").stripSuffix("]") else replaceParens)}
                            .mkString(unionTypeOperator)
                       //+ ") " 
                         + (if(useDotty) ")" else ")#check") + "[" + y.GetTypeStr.getString + "]"
                         ))

           val optionalImplicitsInputs = (optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
           .map(y =>
           //TODO: extract
            "ev" + y.GetTypeStr.getString + ":" + (if(useDotty) "(" else "(UNil TypeOr ") + typeStringMap(y.GetTypeStr.getString).map{ a =>
              val replaceParens = a.replaceAll("\\(", "[").replaceAll("\\)", "]")
              (if(replaceParens.contains("Tensor[")) replaceParens.stripPrefix("Tensor[").stripSuffix("]") else replaceParens)}
                            .mkString(unionTypeOperator)
                       //+ ") " 
                         + (if(useDotty) ")" else ")#check") + "[" + y.GetTypeStr.getString + "]"
                        ))


           val implicitsOutputs = (outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
           .map(y =>
           //TODO: extract
            "ev" + y.GetTypeStr.getString + ":" + (if(useDotty) "(" else "(UNil TypeOr ") + typeStringMap(y.GetTypeStr.getString).map{ a =>
              val replaceParens = a.replaceAll("\\(", "[").replaceAll("\\)", "]")
              (if(replaceParens.contains("Tensor[")) replaceParens.stripPrefix("Tensor[").stripSuffix("]") else replaceParens)}
                            .mkString(unionTypeOperator)
                       //+ ") " 
                         + (if(useDotty) ")" else ")#check") + "[" + y.GetTypeStr.getString + "]"
                        ))



      "\n  def " + x._1 + x._2(z)._2.toString + (if(useFS)"Free" else "") +
//      (if(x._2(z)._2 < maxSinceVersion) x._2(z)._2.toString else "") +
      (if (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0) "[" else "") +
        (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
        .map(y =>
        //TODO: HIGH PRIORITY: Implement specialization via Spire where possible 
        //"@sp(" +  
        y.GetTypeStr.getString + (if(useDotty) " <: " else " : ") + "Numeric:ClassTag:Field"
        ) ++
        optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
          .map(y =>
            //"@sp(" + 
              y.GetTypeStr.getString + (if(useDotty) " <: " else " : ") +  "Numeric:ClassTag:Field"
                      ) ++
        outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
        .map(y =>
           //"@sp(" +  
                 y.GetTypeStr.getString + (if(useDotty) " <: " else " : ") + "Numeric:ClassTag:Field"
                      ) 
        ).distinct.mkString(",") +
      (if (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0) ", J <: XInt]" else "[J <:XInt]") +
      "(" + 
      "name: String" +
      (if (requiredInputs(z).size > 0 || optionalInputs(z).size > 0) "," else "") +
      requiredInputs(z)
        .map(y =>
            y.GetName.getString.replaceAll("var", "someVar") 
                       + ": " +
                       (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString) && typeStringMap(y.GetTypeStr.getString).exists(_.contains("Tensor"))) "Tensor[" + y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long, J]").replaceAll("tensor\\(float\\)","Tensor[Float]") + ", J]" else  y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long, J]").replaceAll("tensor\\(float\\)","Tensor[Float, J]"))
            + ", " + y.GetName.getString + "name: String"
            )
        .mkString(", ") +
      (if (requiredInputs(z).size > 0 && optionalInputs(z).size > 0) "," else "") +
      optionalInputs(z)
        .map(y =>
          y.GetName.getString
            .replaceAll("var", "someVar")
            .replaceAll("shape", "shapeInput") 
            + ": " + "Option[" +
                       (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString) && typeStringMap(y.GetTypeStr.getString).exists(_.contains("Tensor"))) "Tensor[" + y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long, J]").replaceAll("tensor\\(float\\)","Tensor[Float]") + ", J]" else  y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long, J]").replaceAll("tensor\\(float\\)","Tensor[Float, J]")) + 
                         "] = None"
            )
        .mkString(", ") +
      (if (attributesStrings(z).size > 0 && (requiredInputs(z).size + optionalInputs(z).size) > 0)
         "," + attributesStrings(z)
       else "") +
      ")\n" +
      (if((requiredImplicitsInputs ++ optionalImplicitsInputs ++ implicitsOutputs).size > 0) "(implicit " else "") + (requiredImplicitsInputs ++ optionalImplicitsInputs ++ implicitsOutputs).distinct.mkString(",") +  (if((requiredImplicitsInputs ++ optionalImplicitsInputs ++ implicitsOutputs).size > 0) ")" else "") +
      "    : " + //TODO: HIGH PRIORITY: invoke def on parent trait - without, cannot re-use backends
      (if(useFS) "FS[" else "") + "(" + 
      outputs(z)
      .map(y =>
        "" + (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString) && typeStringMap(y.GetTypeStr.getString).exists(_.contains("Tensor"))) "Tensor[" + y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long, J]").replaceAll("tensor\\(float\\)","Tensor[Float]") + ", J]" else  y.GetTypeStr.getString.replaceAll("tensor\\(int64\\)","Tensor[Long, J]").replaceAll("tensor\\(float\\)","Tensor[Float, J]")) 
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
    (if(useFS) "import cats.free.Free\n" else "") +
    (if(useFS) "import cats.free.FreeApplicative\n" else "") +
    (if(useFS) "import cats.effect.IO\n" else "") +
    "import scala.language.higherKinds\n" + 
    "import scala.{specialized => sp}\n" +
    "import spire.math.UByte\n" +
    "import spire.math.UShort\n" +
    "import spire.math.UInt\n" +
    "import spire.math.ULong\n" +
    "import spire.math.Complex\n" +
    "import spire.math.Numeric\n" +
    "import spire.implicits._\n" +
    "import spire.algebra.Field\n" +
    "import scala.reflect.ClassTag\n" +
    (if(useFS) "import onnx._\n" else "") +
    "import singleton.ops._\n\n" + 
//    "import scala.language.higherKinds\n\n" +
    "package" + (if(useFS) " object" else " object") + " onnx" +  (if(useFS) "Free " else " ") +
    "{\n" +
 (if(useFS) "" else "type |:"  + "[+A1, +A2] = Either[A1, A2]\n") +
    //TODO: Replace with Tensor class from ... Tensorics?
    (if(useFS) "" else "  type Tensor[U, J <: XInt] = Tuple2[Seq[U], Seq[J]]\n") + 
    //TODO: HIGH PRIORITY: Defer effect choice - WIP
//    (if(!useFS) "" else "type G[A] = IO[A]\n") +
//    (if(!useFS) "" else "type Par[F[_], A] = FreeApplicative[F, A]\n") +
//    (if(!useFS) "" else "final type FS[A] = Par[G, A]\n") + 
      //"final type FS[A] = FreeS[F, A]\n" ) +
//    (if(!useFS) "" else  "type FreeS[F[_], A] = " + 
//      (if(useDotty) "Free[[B] => FreeApplicative[F, B], A]\n" else "Free[FreeApplicative[F, ?], A]\n")
//      ) +
//    (if(!useFS) "" else "final type FSSeq[A] = FreeS[G, A]\n") +
    (if(useFS) "" else "  trait Operator\n") +
    (if(useFS) "" else "trait Graph\n") + //TODO: something with Graph
//    (if(useFS) "" else typeStrings) + "\n" +
//    "}\n" +
    (if(useFS || useDotty) "" else """object UnionType {

      trait inv[-A] {}

      sealed trait OrR {
        type L <: OrR
        type R
        type invIntersect
        type intersect
      }

      sealed class TypeOr[A <: OrR, B] extends OrR {
        type L = A
        type R = B

        type intersect = (L#intersect with R)
        type invIntersect = (L#invIntersect with inv[R])
        type check[X] = invIntersect <:< inv[X]
      }

      object UNil extends OrR {
        type intersect = Any
        type invIntersect = inv[Nothing]
      }
      type UNil = UNil.type

    }
    """
    ) +
    (if(useDotty) "" else
    """
    import UnionType._
    """
    ) +
    (if(useFS) "@free " else "")  +
    "trait DataSource" + (if(useFS) "Free extends DataSource" else "") + " {\n" +
    "  def inputData" + (if(useFS) "Free" else "") + "[" + inputTypes + ", J <: XInt]" + checkedTypes + ": " +
    (if(useFS) "FS[" else "") +
    "Tensor[T, J]" + (if(useFS) "]" else "") +"\n" +
    "  def getParams" + (if(useFS) "Free" else "") + "[" + inputTypes + ", J <: XInt](name: String)" + checkedTypes + ": " +
    (if(useFS) "FS[" else "") +
    "Tensor[T, J]" + (if(useFS) "]" else "") +"\n" +
    "  def getAttributes" + (if(useFS) "Free" else "") + "[" + inputTypes + ", J <: XInt](name: String)" + checkedTypes + ": " +
    (if(useFS) "FS[" else "") +
    "Tensor[T, J]" + (if(useFS) "]" else "") +"\n" +
    "}\n" +
    traitStrings +
    "}\n"

  def generate(): Unit = {
    println(fullSource)
//    val onnxSource = fullSource.parse[Source].get
//
//val wrote = Files.write(path, onnxSource.syntax.getBytes("UTF-8"));
 val wrote = Files.write(path, fullSource.getBytes("UTF-8"))
  }

  generate()
}
