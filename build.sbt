lazy val root = (project in file(".")).
settings(
  inThisBuild(List(
    organization := "org.emergentorder",
    scalaOrganization := "org.scala-lang",
    scalaVersion := "2.12.8",
    crossScalaVersions := Seq("2.11.12","2.12.8", "2.13.0-M5"),
    version      := "0.1.0-SNAPSHOT"
  )),
  name := "onnx-freestyle-algebra-generator",
  resolvers += Resolver.mavenLocal,
  resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
  resolvers += Resolver.jcenterRepo,
  publishArtifact in packageDoc := false,
//  addCompilerPlugin("org.spire-math" % "kind-projector" % "0.9.7" cross CrossVersion.binary),

//addCompilerPlugin("org.scalameta" % "paradise" % "3.0.0-M11" cross CrossVersion.full),
// scalacOptions += "-Xplugin-require:macroparadise",
//  scalacOptions in (Compile, console) ~= (_ filterNot (_ contains "paradise")), // macroparadise plugin doesn't work in repl yet.
  scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation", "-Ywarn-unused-import", "-Ywarn-unused:locals,privates"),
    libraryDependencies ++= Seq( 
//      "org.bytedeco" % "javacpp" % "1.4.5-SNAPSHOT",
        "org.bytedeco" % "onnx-platform" % "1.6.0-1.5.3-SNAPSHOT",
//      "org.scalatest" %% "scalatest" % "3.0.5-M1" % Test,
      "org.typelevel" % "spire_2.12" % "0.16.0",
      "org.typelevel" % "cats-core_2.12" % "1.3.1",
      "org.typelevel" % "cats-free_2.12" % "1.3.1",
      "org.typelevel" % "cats-effect_2.12" % "1.0.0",
      "com.typesafe.scala-logging" %% "scala-logging" % "3.9.0",
//      "org.scalameta" % "scalameta_2.12" % "4.0.0-M11",
      "eu.timepit" %% "singleton-ops" % "0.3.0",
      "ch.qos.logback" % "logback-classic" % "1.2.3",
//      "com.github.pureconfig" %% "pureconfig" % "0.9.1",
      "io.frees" %% "frees-core" % "0.8.2"
       ),
//    scalafixSettings,
//    wartremoverErrors ++= Warts.allBut(Wart.PublicInference, Wart.ToString),
//    wartremoverExcluded += baseDirectory.value / "src" / "main" / "scala" / "Float16.scala",
    //TODO: exclude only complaint for default arguments
//    wartremoverExcluded += baseDirectory.value / "src" / "main" / "scala" / "ONNX.scala",
//    wartremoverExcluded += baseDirectory.value / "src" / "main" / "scala" / "ONNXAlgebraFree.scala"
)
