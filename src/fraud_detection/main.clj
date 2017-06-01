(ns fraud-detection.main
  (:require [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))

(def cli-options
  [["-f" "--force-gpu value" "Force GPU to be used"
    :id :force-gpu?
    :parse-fn #(Boolean/parseBoolean %)
    :default true]
   [ "-l" nil
    :long-opt "--live-updates"
    :id :live-updates?
    :default false]])

(defn -main
  [& args]
  (println "Running!")
  (let [argmap (parse-opts args cli-options)]
    (require 'fraud-detection.core)
    ((resolve 'fraud-detection.core/train-forever-uberjar)
     (:options argmap))))
