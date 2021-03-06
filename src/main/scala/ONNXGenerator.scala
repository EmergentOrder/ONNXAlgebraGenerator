/*
 * ONNXGenerator
 * Copyright (c) 2018,2019 Alexander Merritt
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

object ONNXGenerator extends App {
//TODO: Grab doc strings and put them in scaladoc
  //TODO: Enforce shape constraints - using dependent types via singleton and higher-kinded
//TODO: Use numsca for Tensor[Doubles only] ?  or tensorflow_scala[Generic, but not typed by shape] 
//or MXNet[ supprots Float16,Float32,Float64,Int32,UInt8, but most operators Float32 and 64 only] or Compute.scala[Float only, others on roadmap] or none
// LATEST: Use Tensorics, create Scala wrapper
// "Primitive numeric, string, and Boolean types MUST be used as elements of tensors."
// "Version numbers can be used as a simple number, or used to encode semantic versions. If using semver, the convention is to use the two most significant bytes for the major number, the next two bytes for the minor number, and the least significant four bytes for the build/bugfix number. When using semver versioning, at least one of the major/minor numbers MUST be non-zero."
//
//TODO: Be explicit about IR VERSION

//Tensor Shape-
//Each size in the list MUST be expressed as an integral value or as a "dimension variable," a string denoting that the actual size of the dimension is not statically constrained to a particular number. This is useful for declaring interfaces that care about the number of dimensions, but not the exact size of each dimension.
//Deal with  Tensor of unknown dimensionality
//The emptry string "", when used as a dimension name, denotes a single dimension of any cardinality. The string "*", when used as a dimension name, denotes zero or more dimensions of unknown cardinality.

//Shapes MAY be defined using a combination of integers and variables.

//Extensible computation graph model
//An implementation MAY extend ONNX by adding operators expressing semantics beyond the standard set of operators that all implementations MUST support. The mechanism for this is adding operator sets to the opset_import property in a model that depends on the extension operators.
//


//Each operator used within a graph MUST be explicitly declared by one of the operator sets imported by the model.
val useZIO = false
  val useDotty = false
  val unionTypeOperator = "#or"

  //Missing: Non-numeric, Boolean and String
//  val checkedTypes = (if(useDotty) "(" else "(implicit ev:(UNil TypeOr ") + "Float16" + unionTypeOperator + "Float" + unionTypeOperator + "Double" + unionTypeOperator + "Byte" + unionTypeOperator + "Short" + unionTypeOperator + "Int" + unionTypeOperator + "Long" + unionTypeOperator + "UByte" + unionTypeOperator + "UShort" + unionTypeOperator + "UInt" + unionTypeOperator + "ULong" + unionTypeOperator + "Complex[Float]" + unionTypeOperator + "Complex[Double]" + (if(useDotty) ")" else ")#check[T])")

  val inputTypes = "T " + ( ": ")  + "Numeric:ClassTag"

  @SuppressWarnings(Array("org.wartremover.warts.Equals"))
  implicit final class AnyOps[A](self: A) {
    def ===(other: A): Boolean = self == other
  }

  val path = Paths.get("src/main/scala/ONNX" + (if(useZIO) "ZIO" else "") + ".scala");

  def replaceTypeStrings(s: String) = s.replaceAll("uint64", "ULong")
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


  val attrTypeMap = Map(org.bytedeco.onnx.AttributeProto.UNDEFINED ->"Undefined",
                      org.bytedeco.onnx.AttributeProto.FLOAT -> "Float",
                      org.bytedeco.onnx.AttributeProto.INT -> "Int",
                      org.bytedeco.onnx.AttributeProto.STRING -> "String",
                      org.bytedeco.onnx.AttributeProto.TENSOR -> "Tensor",
                      org.bytedeco.onnx.AttributeProto.GRAPH -> "Graph",
                      org.bytedeco.onnx.AttributeProto.FLOATS -> "Array[Float]",
                      org.bytedeco.onnx.AttributeProto.INTS -> "Array[Int]",
                      org.bytedeco.onnx.AttributeProto.STRINGS -> "Array[String]",
                      org.bytedeco.onnx.AttributeProto.TENSORS -> "Array[Tensor]",
                      org.bytedeco.onnx.AttributeProto.GRAPHS -> "Array[Graph]",
                      org.bytedeco.onnx.AttributeProto.SPARSE_TENSOR -> "SparseTensor",
                      org.bytedeco.onnx.AttributeProto.SPARSE_TENSORS -> "Array[SparseTensor]")

//  val loaded =
//    org.bytedeco.javacpp.Loader.load(classOf[org.bytedeco.onnx])

  val schemas = org.bytedeco.onnx.OpSchemaRegistry.get_all_schemas_with_history
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
              
                  case s if s.startsWith("tensor(") =>  {
                    val a = s
                      (typeConstraintParam.type_param_str.getString, replaceTypeStrings(a.capitalize)    
            )
                  }

        case s => {
            (typeConstraintParam.type_param_str.getString, replaceTypeStrings(s.capitalize.replaceAll("\\(", "[").replaceAll("\\)", "]").replaceAll("map", "Map"))
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

        .toMap //+ ("tensor(int64)" -> IndexedSeq("Tensor[Long]")) + ("tensor(float)" -> IndexedSeq("Tensor[Float]"))

   



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
          .replaceAll("scale", "scaleAttr") + " : " + (
                                                         "Option[") + "(" + result._2
          .replaceAll("Tensor", "Tensor[T]") +  //Shouldn't need to do this; the type constraints are not encoded correctly on ONNX side. See hack later to replace for ConstantOfShape
          ")" +
          (if (required) "]" else "] = None")
        str
      }
      .mkString(",")
    }

    //TODO: Handle variadic outputs
    //TODO: Handle initializers
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

    val variadicInputs = x._2.map{y =>
      (0 until y._3.size.toInt)
      .map(z => y._3.get(z))
      .filter(z => z.GetOption === 2)
    }

  
    //TODO: Handle optional outputs
    //"Each node referring to an operator with optional outputs MUST provide a name for each output that is computed and MUST NOT provide names for outputs that are not computed."
    //TODO: empty string name means input or output is optional and unspecified


    val outputs = x._2.map{y =>
      (0 until y._4.size.toInt)
      .map(z => y._4.get(z))
    }

    def buildTypeStrings(in: IndexedSeq[org.bytedeco.onnx.OpSchema.FormalParameter], inImplicit: IndexedSeq[String]) = {
       (in.filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
         .zip(inImplicit)
        .map(y =>
        "@sp " +
        y._1.GetTypeStr.getString + ( " : ") + "Numeric:ClassTag" //TODO: Don't add numeric if not required to be numeric
        )
      )
    }

    val maxSinceVersion = (x._2.map(z => z._2) foldLeft 0)(Math.max)

        val beginString = "trait " + x._1 + 
          (if(useZIO) "ZIO" else "") + " extends Operator" + " {\n"


        def generateDefStringSig(s: org.bytedeco.onnx.OpSchema.FormalParameter) = {
           ( "ev" + s.GetTypeStr.getString + ":" + "Contains[" + s.GetTypeStr.getString + ", Union[") + typeStringMap(s.GetTypeStr.getString).map{ a =>
              val replaceParens = (a).replaceAll("\\(", "[").replaceAll("\\)", "]")
              (if(replaceParens.contains("Tensor[")) replaceParens.stripPrefix("Tensor[").stripSuffix("]") else replaceParens)}
                            .distinct
                            .mkString("]" + unionTypeOperator + "[") +
                              ("]#or[UNil]#create]")
        }

        val defStrings = (0 until 
          requiredInputs.size).map {z =>
           val requiredImplicitsInputs = (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
           .map(y => generateDefStringSig(y)))

           val optionalImplicitsInputs = (optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
           .map(y =>  generateDefStringSig(y)))

           val variadicImplicitsInputs = (variadicInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
           .map(y =>  generateDefStringSig(y)))

           val implicitsOutputs = (outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString))
           .map(y =>  generateDefStringSig(y)))

       val allImplicits = (requiredImplicitsInputs ++ optionalImplicitsInputs ++ variadicImplicitsInputs ++ implicitsOutputs).distinct.mkString(",").replaceAll("tensor", "Tensor")


      def processInput(someInput: scala.collection.immutable.IndexedSeq[org.bytedeco.onnx.OpSchema.FormalParameter], optional: Boolean, variadic: Boolean) = {
        someInput.map(y =>
          y.GetName.getString
            .replaceAll("var", "someVar")
            .replaceAll("shape", "shapeInput")
            + ": " + (if(variadic) "Seq[" else "") + "Option[" +
                       (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString) && typeStringMap(y.GetTypeStr.getString).exists(_.contains("Tensor"))) "Tensor[" + y.GetTypeStr.getString.replaceAll("tensor\\(string\\)", "Tensor[String]").replaceAll("tensor\\(int64\\)","Tensor[Long]").replaceAll("tensor\\(float\\)","Tensor[Float]") + "]" else  y.GetTypeStr.getString.replaceAll("tensor\\(string\\)", "Tensor[String]").replaceAll("tensor\\(int64\\)","Tensor[Long]").replaceAll("tensor\\(float\\)","Tensor[Float]")) +
                         "]" + (if(variadic) "]" else "") + (if(optional && !variadic) " = None" else "") // + (if(variadic) "" else ", " + y.GetName.getString + "name: Option[String]" + (if(optional) " = None" else "") )
            )
//        .map(y => if(optional) y.replaceAll("shape", "shapeInput") else y)
//        .mkString(", ")
      }


      def processOutputs = {
        outputs(z)
      .map(y =>
        "" + (if(typeStringMap.exists(_._1 === y.GetTypeStr.getString) && typeStringMap(y.GetTypeStr.getString).exists(_.contains("Tensor"))) "Tensor[" + y.GetTypeStr.getString.replaceAll("tensor\\(string\\)", "Tensor[String]").replaceAll("tensor\\(int64\\)","Tensor[Long]").replaceAll("tensor\\(float\\)","Tensor[Float]") + "]" else  y.GetTypeStr.getString.replaceAll("tensor\\(string\\)", "Tensor[String]").replaceAll("tensor\\(int64\\)","Tensor[Long]").replaceAll("tensor\\(float\\)","Tensor[Float]")) 
        )
      }

        val typeStrings = (buildTypeStrings(requiredInputs(z), requiredImplicitsInputs) ++
         buildTypeStrings(optionalInputs(z), optionalImplicitsInputs) ++
         buildTypeStrings(variadicInputs(z), variadicImplicitsInputs) ++
         buildTypeStrings(outputs(z), implicitsOutputs)            
        )

        val requiredProcessedInputs = processInput(requiredInputs(z), false, false)
        val optionalProcessedInputs = processInput(optionalInputs(z), true, false)
        val variadicProcessedInputs = processInput(variadicInputs(z), false, true)

        val allProcessedInputs = requiredProcessedInputs ++ optionalProcessedInputs ++ variadicProcessedInputs

      "\n  def " + x._1 + x._2(z)._2.toString + (if(useZIO)"ZIO" else "") +
//      (if(x._2(z)._2 < maxSinceVersion) x._2(z)._2.toString else "") +
      (if (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 ||variadicInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0) "[" else "") +
        typeStrings.distinct.mkString(",") +
      (if (requiredInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || optionalInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || variadicInputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0 || outputs(z).filter(y => typeStringMap.exists(_._1 === y.GetTypeStr.getString)).size > 0) "]" else "") +
      "(" + 
      "name: String" +
      (if (requiredInputs(z).size > 0 || optionalInputs(z).size > 0 || variadicInputs(z).size > 0 || attributesStrings(z).size > 0) "," else "") +
      (if (x._1.contains("ConstantOfShape")) attributesStrings(z).replaceAll("Tensor\\[T\\]", "Tensor[T2]") else attributesStrings(z)) +
      (if (attributesStrings(z).size > 0 && requiredInputs(z).size > 0) "," else "") +
      requiredProcessedInputs.mkString(", ") +
      (if ((requiredInputs(z).size > 0 || attributesStrings(z).size > 0) && optionalInputs(z).size > 0) "," else "") +
      optionalProcessedInputs.mkString(", ") + 
      (if (variadicInputs(z).size > 0 && (requiredInputs(z).size + optionalInputs(z).size + attributesStrings(z).size) > 0)
            ","
       else "") + variadicProcessedInputs.mkString(", ") +
      ")\n" +
      (
      (if((requiredImplicitsInputs ++ optionalImplicitsInputs ++ variadicImplicitsInputs ++ implicitsOutputs).size > 0) "(implicit " else "") + allImplicits +  (if((requiredImplicitsInputs ++ optionalImplicitsInputs ++ variadicImplicitsInputs ++ implicitsOutputs).size > 0) ")" else "")
      ) +
      "    : " + 
      (if(useZIO) "Task[" else "") + "(" + 
      processOutputs(0) + (if(useZIO) "]" else ")") +"\n" +
//Body
" = {\n" + "val map: Map[String, Any] = Map(" + attributesStrings(z).split(",").filter(!_.isEmpty).map{ i => "\"" + i.split(":")(0).trim() + "\"" + " -> " + i.split(":")(0) + "\n"}.mkString(",") +
        ")\n" +
        "val allInputs = (" + (requiredInputs(z).map{i =>  i.GetName.getString.replaceAll("var", "someVar")} ++ optionalInputs(z).map{i => i.GetName.getString.replaceAll("var", "someVar")} ++ (0 until (9 - (optionalInputs(z).size + requiredInputs(z).size))).map{t => if(variadicInputs(z).size > 0) { variadicInputs(z).map{i => i.GetName.getString.replaceAll("var", "someVar") + ".lift(" + t.toString + ").flatten" }.mkString(",")} else { "None : Option[Any]"}}).map(i => i.replaceAll("shape", "shapeInput")).mkString(",") + ")" + 
        "\n" + 
        "(callOp" +
        "[" + (allProcessedInputs.map(i => i.split(":")(1).split("=")(0)) ++ (0 until (9 - allProcessedInputs.size)).map(_ => if(variadicInputs(z).isEmpty){"Any"} else {variadicProcessedInputs.map(i => i.split(":")(1).split("=")(0)).mkString(",")}) ++ processOutputs ++ (0 until (9 - processOutputs.size)).map(_ => "Any")).mkString(",").replaceAll("Seq\\[", "").replaceAll("\\]\\]\\]", "\\]\\]") +"]" +
        "(name,\"" + x._1 + "\"," + "allInputs, map))" +"\n}"
          }.distinct.mkString("\n")

      val endString = "\n}"

      val traitString = beginString + defStrings + endString

    (traitString, typeStringMap)
  }

  val flattenedTypeStringsMap =
    traitStringsAndTypeStrings.map(x => x._2).flatten.toMap

  val traitStrings: String = traitStringsAndTypeStrings
    .map(x => x._1)
    .toSeq
    .sorted
    .filter(x => !x.contains("ATen"))
    .filter(a => a.contains("def")) 
    .mkString("\n")


  val fullSource = "package org.emergentorder\n\n" +
    (if(useZIO) "import zio.Task\n" else "") +
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
    "import org.bytedeco.onnx.ModelProto\n" + 
    (if(useZIO) "import onnx._\n" else "") +
//    "import scala.language.higherKinds\n\n" +
    "package" + (if(useZIO) " object" else " object") + " onnx" +  (if(useZIO) "ZIO " else " ") +
    "{\n" +
    (if(useZIO) "" else 
  """
  type ![A] = A => Nothing
  type !![A] = ![![A]]

  trait Disjunction[T] {
    type or[S] = Disjunction[T with ![S]]
    type create = ![T]
  }

  type Union[T] = {
    type or[S] = Disjunction[![T]]#or[S]
  }

  type Contains[S, T] = !![S] <:< T

  type UNil

  trait Dim

  sealed trait Axes

  sealed trait Scalar extends Axes
  sealed trait Vec[T <: Dim] extends Axes
  sealed trait Mat[T <: Dim, U <: Dim] extends Axes
  sealed trait Tuple3OfDim[T <: Dim, U <: Dim, V <: Dim] extends Axes

  type TypesafeTensor[T, A <: Axes] = Tuple2[Array[T], Array[Int]]

  type Tensor[T] = TypesafeTensor[T, Axes]
  type SparseTensor[T] = Tensor[T]

  type XInt = Int with Singleton

  object TensorFactory {
    def getTensor[T](data: Array[T], t: Array[Int]): Tensor[T] = {
      require(data.size == t.foldLeft(1)(_ * _))
      (data, t)
    }
   }
  """ + "\n") +
    (if(useZIO) "" else """
  
    trait Operator {
    def callOp[
        T: ClassTag,
        T1: ClassTag,
        T2: ClassTag,
        T3: ClassTag,
        T4: ClassTag,
        T5: ClassTag,
        T6: ClassTag,
        T7: ClassTag,
        T8: ClassTag,
        T9: ClassTag,
        T10: ClassTag,
        T11: ClassTag,
        T12: ClassTag,
        T13: ClassTag,
        T14: ClassTag,
        T15: ClassTag,
        T16: ClassTag,
        T17: ClassTag
    ](
        name: String,
        opName: String,
        inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8],
        //    outName: String,
        attrs: Map[String, Any]
    ): (T9)
  }

  abstract class Model(onnxBytes: Array[Byte]) extends Operator{
    def fullModel[
      T: ClassTag,
      T1: ClassTag,
      T2: ClassTag,
      T3: ClassTag,
      T4: ClassTag,
      T5: ClassTag,
      T6: ClassTag,
      T7: ClassTag,
      T8: ClassTag,
      T9: ClassTag,
      T10: ClassTag,
      T11: ClassTag,
      T12: ClassTag,
      T13: ClassTag,
      T14: ClassTag,
      T15: ClassTag,
      T16: ClassTag,
      T17: ClassTag
    ](
      inputs: Tuple9[T, T1, T2, T3, T4, T5, T6, T7, T8]
  ): (T9)
  }

      """ ) +
    (if(useZIO) "" else "  trait Graph\n") + //TODO: something with Graph
//    (if(useZIO) "" else typeStrings) + "\n" +
//    "}\n" +
    "trait DataSource" + (if(useZIO) "ZIO " else "") + " {\n" +
    "  def getParams" + (if(useZIO) "ZIO" else "") + "[" + inputTypes  + "](name: String)" + ("") + ": " +
    (if(useZIO) "Task[" else "") +
    "Tensor[T]" + (if(useZIO) "]" else "") +"\n" +
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
