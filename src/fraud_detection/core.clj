(ns fraud-detection.core
  :require [clojure.java.io :as io]
           [clojure.string :as string]
           [clojure.data.csv :as csv]
           [clojure.core.matrix :as mat]
           [cortex.nn.layers :as layers]
           [cortex.nn.network :as network]
           [cortex.nn.execute :as execute]
           [cortex.optimize :as opt]
           [cortex.optimize.adam :as adam]
           [cortex.tree :as tree]
           [cortex.experiment.classification :as classification]
           [cortex.experiment.train :as train]
           [cortex.experiment.train :as experiment-train])

; 492 positives, 284807 total
(defonce credit-data (with-open [infile (io/reader "resources/creditcard.csv")]
                    (rest (doall (csv/read-csv infile)))))
(defonce data (mat/matrix (map #(mapv read-string %) (mapv drop-last credit-data))))
(defonce labels (mat/matrix (map read-string (map last credit-data))))


(defn create-dataset
  ([] (create-dataset 0.8))
  ([train-split]
    (let [dataset (shuffle (mapv (fn [d l] {:data d :label l}) data labels))
          train-ds (take (int (* (count dataset) train-split)) dataset)
          test-ds (drop (int (* (count dataset) train-split)) dataset)] ; [{:data [1.3, 2.9...], :label 0}...]
      [train-ds test-ds])))


(defn network-description
  [width height]
  (->> [(layers/input width height 1 :id :data) ;width, height, channels, args
        (layers/linear 15) ; num-output & args
        (layers/linear 10)
        (layers/linear 2)
        (layers/softmax :id :labels)]
       (network/linear-network)))

(defn random-forest
  [num-trees]
  (tree/random-forest data labels {:n-trees num-trees
                                   :split-fn tree/best-splitter}))


(defn train-forever
  [& {:keys [batch-size] :or {batch-size 128}}]
  (println "Training forever")
  (let [[train-ds test-ds] create-dataset
        network (network-description (count ((first train-ds) :data)) 1)]
    (experiment-train/train-n network
                              train-ds test-ds
                              :optimizer (adam/adam)
                              :batch-size batch-size)))

(defn train-forever-uberjar
  ([] (train-forever-uberjar {}))
  ([argmap]
    (train-forever))) ; train-forever argmap
