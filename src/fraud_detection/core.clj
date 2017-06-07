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
   :batch-size        100
   :epoch-count       100
   :epoch-size        200000})

(defn label->vec
"One-hot vector transformations (label->vec [:a :b :c :d] :b) => [0 1 0 0]"
  [class-names label]
  (let [num-classes (count class-names)
        src-vec (vec (repeat num-classes 0))]
    (assoc src-vec (.indexOf class-names label) 1)))

(defn vec->label
  "Map vector of probabilities to original class (vec->label [:a :b :c :d] [0.1 0.2 0.6 0.1] => :c)"
  [class-names label-vec]
  (let [max-idx (util/max-index label-vec)]
    (nth class-names max-idx)))

;; "Read input csv and create a vector of maps {:data [...] :label [..]}, where each map represents one instance"
(defonce create-dataset
  (memoize
    (fn []
      (let [credit-data (with-open [infile (io/reader orig-data-file)]
                          (rest (doall (csv/read-csv infile))))
            data (mapv #(mapv read-string %) (map drop-last credit-data))
            labels (mapv #(label->vec [0 1] (read-string %)) (map last credit-data))
            dataset (mapv (fn [d l] {:data d :label l}) data labels)]
        dataset))))


;; "Split dataset into train/test sets with an equal proportion of positive examples in each
;; (as a way of reducing testing variation in a highly imbalanced dataset)"
(defonce get-train-test-dataset
  (memoize
    (fn []
      (let [dataset (shuffle (create-dataset))
            {positives true negatives false} (group-by #(= (:label %) [0 1]) dataset)
            test-pos-amount (int (* (count positives) (/ (:test-ds-size params) (count dataset)))) ;86
            test-neg-amount (- (:test-ds-size params) test-pos-amount)
            test-set (into [] (concat (take test-pos-amount positives) (take test-neg-amount negatives)))
            train-set (into [] (concat (drop test-pos-amount positives) (drop test-neg-amount negatives)))]
        [train-set test-set]))))


;; for random forest
; (defonce get-train-test-matrices
;   (memoize
;     (fn [type]
;       (let [[train-ds test-ds] (get-train-test-dataset)
;             train-data (mat/matrix (map #(:data %) train-ds))
;             train-labels (mat/matrix (map #(vec->label [0 1] (:label %)) train-ds))
;             test-data (mat/matrix (map #(:data %) test-ds))
;             test-labels (mat/matrix (map #(vec->label [0 1] (:label %)) test-ds))]
;         (if (= type :train)
;           [train-data train-labels]
;           [test-data test-labels])))))



(defn calc-min-dist
  "Calculate min distance between feature vectors for positive and negative samples

  Result: Elapsed time: 1696027.647549 msecs  │  3.598395571684021"
  []
  (let [dataset (shuffle (create-dataset))
        {positives true negatives false} (group-by #(= (:label %) [0 1]) dataset)
        pos-data (mat/matrix (map #(:data %) positives))
        neg-data (mat/matrix (map #(:data %) negatives))
        dists (for [p pos-data n neg-data] (mat/distance p n))]
    (time (apply min dists))))


(defn add-rand-vec
  "Take vector v and add a random vector with elements between 0 and epsilon"
  [v epsilon]
  (let [len (count v)
        randv (take len (repeatedly #(- (* 2 (rand epsilon)) epsilon)))]
    (mapv + v randv)))

(defn augment-train-ds
  "Takes train dataset and augments positive examples to reach 50/50 balance"
  [orig-train]
  (let [{train-pos true train-neg false} (group-by #(= (:label %) [0 1]) orig-train)
        pos-data (map #(:data %) train-pos)
        num-augments (- (count train-neg) (count train-pos))
        augments-per-sample (int (/ num-augments (count train-pos)))

        augmented-data (apply concat (repeatedly augments-per-sample #(mapv (fn [p] (add-rand-vec p 0.5)) pos-data)))
        augmented-ds (mapv (fn [d] {:data d :label [0 1]}) augmented-data)]
    (shuffle (concat orig-train augmented-ds))))


(def network-description
  [(layers/input (count (:data (first (create-dataset)))) 1 1 :id :data) ;width, height, channels, args
  (layers/linear->relu 15) ; num-output & args
  ;(layers/dropout 0.9)
  ;(layers/linear->relu 10)
  (layers/linear 2)
  (layers/softmax :id :label)])

(comment
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
)


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

;; Testing only purposes
(defn get-true-pos-false-pos
  [y y_hat]
  (let [true-pos-count (mat/esum (metrics/true-positives y y_hat))
        false-pos-count (mat/esum (metrics/false-positives y y_hat))]
    [true-pos-count false-pos-count]))

(defn train
  "Train the network for epoch-count epochs, saving the best results as we go."
  []
  (log "________________________________________________________")
  (log params)
  (log (str network-description))
  (let [context (execute/compute-context)] ; determines context for gpu/cpu training
    (execute/with-compute-context
      context
      (let [[train-orig test-ds] (get-train-test-dataset)
            train-ds (take (:epoch-size params) (shuffle (augment-train-ds train-orig)))
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

                              test-accuracy (loss/evaluate-softmax (map :label test-results) (map :label test-ds))

                              ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

                              train-results (execute/run network (take (:test-ds-size params) train-ds) :context context
                                                                                                 :batch-size (:batch-size params))
                              ;;; train metrics
                              train-actual (vec (map #(vec->label [0 1] %) (map :label (take (count test-ds) train-ds))))
                              train-pred (vec (map #(vec->label [0 1] %) (map :label (take (count test-ds) train-results))))

                              train-precision (metrics/precision train-actual train-pred)
                              train-recall (metrics/recall train-actual train-pred)
                              train-f-beta (f-beta train-precision train-recall)

                              train-accuracy (loss/evaluate-softmax (map :label train-results) (map :label (take (:test-ds-size params) train-ds)))

                              ]

                            (log (str "Epoch: " (inc epoch) "\n"
                                      "Test accuracy: " test-accuracy "              | Train accuracy: " train-accuracy "\n"
                                      "Test precision: " test-precision  "              | Train precision: " train-precision"\n"  ;; "              | Train precision: " train-precision
                                      (if (Double/isNaN test-precision)
                                          (str "[True pos, false pos]: "(get-true-pos-false-pos test-actual test-pred) "\n")
                                          nil)
                                      "Test recall: " test-recall "                | Train recall: " train-recall "\n"       ;; "              | Train recall: " train-recall
                                      "Test F1: " test-f-beta "                | Train F1: " train-f-beta "\n\n"))       ;; "              | Train F1: " train-f-beta

                            (when (> test-f-beta (:test-score @high-score*))
                                  (reset! high-score* {:test-score test-f-beta :train-score 0})
                                  (save network))
                            [network optimizer]))
                [network (:optimizer params)]
                (range (:epoch-count params)))
              (println "Done.")))))
