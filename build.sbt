lazy val root = (project in file(".")).
settings(
  inThisBuild(List(
    organization := "org.emergentorder",
    scalaOrganization := "org.scala-lang",
    scalaVersion := "2.12.4",
    crossScalaVersions := Seq("2.11.12","2.12.4", "2.13.0-M3"),
    version      := "0.1.0-SNAPSHOT"
  )),
  name := "onnx-freestyle-algebra-generator",
  resolvers += Resolver.mavenLocal,
  resolvers += Resolver.jcenterRepo,
addCompilerPlugin("org.scalameta" % "paradise" % "3.0.0-M11" cross CrossVersion.full),
// scalacOptions += "-Xplugin-require:macroparadise",
//  scalacOptions in (Compile, console) ~= (_ filterNot (_ contains "paradise")), // macroparadise plugin doesn't work in repl yet.
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation", "-Ywarn-unused-import", "-Ywarn-unused:locals,privates"),
    libraryDependencies ++= Seq( 
      "org.bytedeco" % "javacpp" % "1.4.1",
      "org.scalatest" %% "scalatest" % "3.0.5-M1" % Test,
      "org.typelevel" %% "spire" % "0.15.0",
      "org.typelevel" %% "cats-core" % "1.1.0",
      "org.typelevel" %% "cats-effect" % "1.0.0-RC",
      "com.typesafe.scala-logging" %% "scala-logging" % "3.9.0",
      "ch.qos.logback" % "logback-classic" % "1.2.3",
      "com.github.pureconfig" %% "pureconfig" % "0.9.1",
      "io.frees" % "frees-core_2.12" % "0.8.0"
       ),
//    scalafixSettings,
    wartremoverErrors ++= Warts.allBut(Wart.PublicInference),
    wartremoverExcluded += baseDirectory.value / "src" / "main" / "scala" / "Float16.scala",
    wartremoverExcluded += baseDirectory.value / "src" / "main" / "scala" / "ONNXAlgebra.scala"
)
