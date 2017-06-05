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
           [cortex.metrics :as metrics]
           [cortex.tree :as tree]
           [cortex.loss :as loss]
           [cortex.util :as util]))


(def orig-data-file "resources/creditcard.csv")
(def log-file "training.log")
(def network-file "trained-network.nippy")

(def params
  {:test-ds-size      50000 ;; total = 284807, test-ds ~= 17.5%
   :optimizer         (adam/adam)   ;; alternately, (adadelta/adadelta)
   :batch-size        200
   :epoch-count       100
   :epoch-size        200000})

(defn label->vec ;; (label->vec [:a :b :c :d] :b) => [0 1 0 0]
 [class-names label]
 (let [num-classes (count class-names)
       src-vec (vec (repeat num-classes 0))]
   (assoc src-vec (.indexOf class-names label) 1)))

(defn vec->label ;; (vec->label [:a :b :c :d] [0.1 0.2 0.6 0.1] => :c)
  [class-names label-vec]
  (let [max-idx (util/max-index label-vec)]
    (nth class-names max-idx)))

(defonce create-dataset
  (memoize
    (fn []
      (let [credit-data (with-open [infile (io/reader orig-data-file)]
                          (rest (doall (csv/read-csv infile))))
            data (mapv #(mapv read-string %) (map drop-last credit-data))
            labels (mapv #(label->vec [0 1] (read-string %)) (map last credit-data))
            dataset (mapv (fn [d l] {:data d :label l}) data labels)]
        dataset))))


(defn make-infinite
  "Create endless stream of shuffled dataset"
  [dataset]
  (apply concat (repeatedly #(shuffle dataset))))

;; "Returns vector of [train-dataset-map, test-dataset-map] where each map has {:data [...], :label [..]}"
(defonce get-train-test-dataset
  (memoize
    (fn []
      (let [dataset (shuffle (create-dataset))
            {positives true negatives false} (group-by #(= (:label %) [0 1]) dataset)
            test-pos-amount (int (* (count positives) (/ (:test-ds-size params) (count dataset)))) ;86
            test-neg-amount (- (:test-ds-size params) test-pos-amount)
            test-set (into [] (concat (take test-pos-amount positives) (take test-neg-amount negatives)))
            train-set (into [] (concat (drop test-pos-amount positives) (drop test-neg-amount negatives)))]
        [(shuffle train-set) (shuffle test-set)]))))

;; for random forest
(defonce get-train-test-matrices
  (memoize
    (fn [type]
      (let [[train-ds test-ds] (get-train-test-dataset)
            train-data (mat/matrix (map #(:data %) train-ds))
            train-labels (mat/matrix (map #(:label %) train-ds))
            test-data (mat/matrix (map #(:data %) test-ds))
            test-labels (mat/matrix (map #(:label %) test-ds))]
        (if (= type :train)
          [train-data train-labels]
          [test-data test-labels])))))


(def network-description
  [(layers/input (count (:data (first (create-dataset)))) 1 1 :id :data) ;width, height, channels, args
  (layers/linear->relu 20) ; num-output & args
  (layers/linear->relu 10)
  (layers/linear 2)
  (layers/softmax :id :label)])

(defn random-forest
  [num-trees]
  (let [[train-data train-labels] (get-train-test-matrices :train)
        rf (tree/random-forest (mat/submatrix train-data 0 [0, 50000]) (mat/submatrix train-labels 0 [0, 50000]) {:n-trees num-trees
                                                     :split-fn tree/best-splitter})]
    (println "Trained random forest")
    rf))

(defn test-random-forest
  []
  (let [rf (random-forest 50)
        [test-data test-labels] (get-train-test-matrices :test)
        pred-labels (map #(tree/forest-classify rf %) test-data)]
    pred-labels
))


(defn log
     "Print data and save to log file"
     [data]
     (spit log-file (str data "\n") :append true) ;; explain when blogging
     (println data))

(defn save
     "Save the network to a nippy file."
     [network]
     (log (str "Saving network to " network-file))
     (util/write-nippy-file network-file network))

(defn sigmoid
  [x]
  (/ 1 (+ 1 (Math/pow (Math/E) (- x)))))


;; false positive ok
;; false negative bad (fraud happens)
;; => emphasize recall
(defn f-beta
  "F-beta score, default uses F1"
  ([precision recall] (f-beta precision recall 1))
  ([precision recall beta]
    (let [beta-squared (* beta beta)]
      (* (+ 1 beta-squared)
         (try                         ;; catch divide by 0 errors
           (/ (* precision recall)
              (+ (* beta-squared precision) recall))
         (catch ArithmeticException e
           0))))))


(def high-score* (atom {:test-score 0 :train-score 0}))

(defn train
  "Train the network for epoch-count epochs, saving the best results as we go."
  []
  (log "________________________________________________________")
  (log params)
  (let [context (execute/compute-context)] ; determines context for gpu/cpu training
    (execute/with-compute-context
      context
      (let [[train-orig test-ds] (get-train-test-dataset)
            train-ds (take (:epoch-size params) (shuffle train-orig))
            network (network/linear-network network-description)]
            (reduce (fn [[network optimizer] epoch]
                        (let [{:keys [network optimizer]} (execute/train network train-ds
                                                                        :context context
                                                                        :batch-size (:batch-size params)
                                                                        :optimizer optimizer)
                              test-results  (execute/run network test-ds :context context
                                                                         :batch-size (:batch-size params))
                              ;;; test metrics
                              test-actual (vec (map #(vec->label [0 1] %) (map :label test-ds)))
                              test-pred (vec (map #(vec->label [0 1] %) (map :label test-results)))

                              test-precision (metrics/precision test-actual test-pred)
                              test-recall (metrics/recall test-actual test-pred)
                              test-f-beta (f-beta test-precision test-recall)

                              ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

                              train-results (execute/run network (take (count test-ds) train-ds) :context context
                                                                                                 :batch-size (:batch-size params))
                              ;;; train metrics
                              train-actual (vec (map #(vec->label [0 1] %) (map :label (take (count test-ds) train-ds))))
                              train-pred (vec (map #(vec->label [0 1] %) (map :label (take (count test-ds) train-results))))

                              train-precision (metrics/precision train-actual train-pred)
                              train-recall (metrics/recall train-actual train-pred)
                              train-f-beta (f-beta train-precision train-recall)

                              ]

                            (log (str "Epoch: " (inc epoch) "\n"
                                      "Test precision: " test-precision "              | Train precision: " train-precision "\n" ;; "              | Train precision: " train-precision
                                      "Test recall: " test-recall "              | Train recall: " train-recall "\n"       ;; "              | Train recall: " train-recall
                                      "Test F1: " test-f-beta "              | Train F1: " train-f-beta "\n\n"))       ;; "              | Train F1: " train-f-beta

                            (when (> test-f-beta (:test-score @high-score*))
                                  (reset! high-score* {:test-score test-f-beta :train-score 0})
                                  (save network))
                            [network optimizer]))
                [network (:optimizer params)]
                (range (:epoch-count params)))
              (println "Done.")))))
