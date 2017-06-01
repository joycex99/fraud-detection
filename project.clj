(defproject fraud-detection "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [thinktopic/cortex "0.9.9-SNAPSHOT"]
                 [org.clojure/clojurescript "1.8.51"]
                 [org.bytedeco.javacpp-presets/cuda "8.0-1.2"]
                 [org.clojure/data.csv "0.1.3"]]

  :main fraud-detection.main
  :aot [fraud-detection.main]
  :uberjar-name "classify-example.jar"

  :clean-targets ^{:protect false} [:target-path
                                    "figwheel_server.log"
                                    "resources/public/out/"
                                    "resources/public/js/app.js"])
