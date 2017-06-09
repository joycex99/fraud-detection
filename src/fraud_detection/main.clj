(ns fraud-detection.main
  (:gen-class))

(defn -main
  [& args]
  (println "Loading our dataset and functions...")
  (require 'fraud-detection.core)
  (println "Loaded! Training:")
  ((resolve 'fraud-detection.core/train)))
