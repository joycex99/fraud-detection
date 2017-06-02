(ns fraud-detection.core
  (:require [clojure.java.io :as io]
           [clojure.string :as string]
           [clojure.data.csv :as csv]
           [clojure.core.matrix :as mat]
           [cortex.nn.layers :as layers]
           [cortex.nn.network :as network]
           [cortex.nn.execute :as execute]
           [cortex.optimize.adadelta :as adadelta]
           [cortex.optimize.adam :as adam]
           [cortex.tree :as tree]
           [cortex.loss :as loss]
           [cortex.util :as util]))

; 492 positives, 284807 total
; (defonce credit-data (with-open [infile (io/reader "resources/creditcard.csv")]
;                     (rest (doall (csv/read-csv infile)))))
; (defonce data (mat/matrix (map #(mapv read-string %) (mapv drop-last credit-data))))
; (defonce labels (mat/matrix (map read-string (map last credit-data))))

(def orig-data-file "resources/creditcard.csv")
(def log-file "training.log")
(def network-file "trained-network.nippy")

(def params
  {:train-test-split  0.8
   :optimizer         (adam/adam)   ;; alternately, (adadelta/adadelta)
   :batch-size        10
   :epoch-count       40
   :epoch-size        160000})

(defn label->vec
 [class-names label]
 (let [num-classes (count class-names)
       src-vec (vec (repeat num-classes 0))
       class-name->index (into {} (map-indexed (comp vec reverse list) class-names))]
   (assoc src-vec (class-name->index label) 1)))

(defonce create-dataset
  (memoize
    (fn []
      (let [credit-data (with-open [infile (io/reader orig-data-file)]
                          (rest (doall (csv/read-csv infile))))
            cropped-credit-data (drop-last 7 credit-data)
            data (mat/matrix (map #(mapv read-string %) (mapv drop-last cropped-credit-data)))
            labels (mat/matrix (map #(label->vec [0 1] (read-string %)) (map last cropped-credit-data)))]
        [data labels]))))

(defn make-infinite
  "Create endless stream of shuffled dataset"
  [dataset]
  (apply concat (repeatedly #(shuffle dataset))))

(defonce get-train-test ; 227845, 56962
  (memoize
    (fn []
      (let [[data labels] (create-dataset)
            dataset (shuffle (mapv (fn [d l] {:data d :label l}) data labels))
            train-ds (take (int (* (count dataset) (:train-test-split params))) dataset)
            test-ds (drop (int (* (count dataset) (:train-test-split params))) dataset)] ; [{:data [1.3, 2.9...], :label 0}...]
        [train-ds test-ds]))))



(def network-description
  [(layers/input (second (mat/shape (first (create-dataset)))) 1 1 :id :data) ;width, height, channels, args
  (layers/linear 15) ; num-output & args
  (layers/linear 10)
  (layers/linear 2)
  (layers/softmax :id :label)])

(defn random-forest
  [num-trees]
  (let [[data labels] (create-dataset)]
    (tree/random-forest data labels {:n-trees num-trees
                                     :split-fn tree/best-splitter})))



(defn log
     "Print data and save to log file"
     [data]
     (spit log-file (str data "\n") :append true) ;; explain when blogging
     (println data))

(defn save
     "Save the network to a nippy file."
     [network]
     (println "Saving network to" network-file)
     (util/write-nippy-file network-file network))

(defn sigmoid
  [x]
  (/ 1 (+ 1 (Math/pow (Math/E) (- x)))))


(def high-score* (atom {:test-score 0 :train-score 0}))

(defn train
  "Train the network for epoch-count epochs, saving the best results as we go."
  []
  (log "______________________________________________")
  (let [context (execute/compute-context)] ; determines context for gpu/cpu training
    (execute/with-compute-context
      context
      (let [[train-ds test-ds] (get-train-test)
            network (network/linear-network network-description)]
            (reduce (fn [[network optimizer] epoch]
                        (let [{:keys [network optimizer]} (execute/train network train-ds
                                                                        :context context
                                                                        :batch-size (:batch-size params)
                                                                        :optimizer optimizer)
                              test-results  (execute/run network test-ds :context context
                                                                         :batch-size (:batch-size params))
                              test-score    (loss/evaluate-softmax (map :label test-results) (map :label test-ds)) ;; change
                              train-results (execute/run network (take (count test-ds) train-ds) :context context
                                                                                                 :batch-size (:batch-size params))
                              train-score   (loss/evaluate-softmax (map :label train-results) (map :label (take (count test-ds) train-ds)))]
                            (log (str "Epoch: " (inc epoch) " | test score: " test-score " | training score: " train-score))
                            (when (> test-score (:test-score @high-score*))
                                  (reset! high-score* {:test-score test-score :train-score train-score})
                                  (save network))
                            [network optimizer]))
                [network (:optimizer params)]
                (range (:epoch-count params)))
              (println "Done.")))))
