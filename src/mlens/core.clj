(ns mlens.core
  (:require [flambo.api :as f]
            [flambo.tuple :as ft]
            [flambo.conf :as conf])
  (:import [org.apache.log4j Level Logger]
           [scala Tuple2]
           [org.apache.spark.api.java.function Function]
           [org.apache.spark.mllib.recommendation ALS MatrixFactorizationModel Rating])) 

;; We don't need to see everything;
(.setLevel (Logger/getRootLogger) Level/WARN)
;; Set up Spark
(def c (-> (conf/spark-conf)
           (conf/master "local[*]")
           (conf/app-name "mlens")))

(def sc (f/spark-context c))

; Spark fns
(f/defsparkfn split-line 
  [line]
  (clojure.string/split line #"\t"))

(f/defsparkfn parse-rating 
  [[user-id product-id rating timestamp]]
  (Rating. (Integer/parseInt user-id)
           (Integer/parseInt product-id)
           (Double/parseDouble rating)))

(f/defsparkfn user-product-tuple
   [rating]
   (ft/tuple (.user rating) 
             (.product rating)))

(f/defsparkfn wrap-rating
    [r] 
    (ft/tuple (ft/tuple (.user r) (.product r))
              (.rating r)))

(f/defsparkfn unscala
  [r]
  (map f/untuple (f/untuple r)))

(f/defsparkfn squared-error
    [r]
    (let [r-vals  (last r)]
      (* (- (first r-vals) (second r-vals))
         (- (first r-vals) (second r-vals)))))

; dataset pre-processing
(def data-source (f/text-file sc "resources/u.data"))

(def data (-> data-source
              (f/map split-line)))

(defn to-ratings
  [d]
  (-> d 
      (f/map parse-rating)))

(def initial-data
  {:iteration 0
   :training (.cache (.sample data false 0.7))
   :testing (.subtract data training)})

(defn default-pre-processing
  [data]
  {:iteration (inc (:iteration data)) 
   :training (.cache (.sample data false 0.7))
   :testing (.subtract data training)})

(defn fit-model
  [training]
  (let [rank 10
        numIterations 20
        model (ALS/train (.rdd (to-ratings training)) rank numIterations 0.01)]
    model))

(defn get-predictions
  [model testing]
  (let [predictions (-> (.toJavaRDD 
                          (.predict model 
                                    (.rdd (to-user-products 
                                            (to-ratings testing)))))
                        (f/map-to-pair wrap-rating))]
    predictions))

(defn evaluate
  [model predictions testin]
  (let [SSE (-> (f/join rates predictions)
                (f/map unscala)
                (f/map squared-error)
                (f/reduce (f/fn [x y] (+ x y))))
        MSE (/ SSE (f/count testing))
        RMSE (Math/sqrt MSE)]
    {:SSE SSE
     :MSE MSE
     :RMSE RMSE}))

(defn default-experiment
  [data]
  (let  [model (fit-model (:training data))
         predictions (get-predictions model (:testing data))
         metrics (evaluate model predictions (:testing data))]
    {:iteration (:iteration data)
     :training (:training data) 
     :testing (:testing data)
     :model model
     :predictions predictions
     :metrics metrics}))

(defn default-post-processing
  [data]
  data)

(def pre-processing default-pre-processing)
(def experiment default-experiment)
(def post-processing default-post-processing)

(def lazy-run-experiment
  ((fn exp [data]
     (let [input (pre-processing data)
           result (default-experiment input)
           next-data (post-processing result)]
     (lazy-seq (cons 
                 result
                 (exp next-data)))))
   initial-data))

(defn -main
  []
  (let [run-count 5
        runs (take run-count lazy-run-experiment)
        avg-train-count (float (/ (reduce + (map #(f/count (:training %)) runs))
                                  run-count)) 
        avg-test-count (float (/ (reduce + (map #(f/count (:testing %)) runs))
                                  run-count)) 
        avg-rmse (float (/ (reduce + (map #(:RMSE (:metrics %)) runs))
                           run-count))
        ]
    (println "---------------------")
    (println "Ratings  :" (f/count data))
    (println "Runs     :" run-count)
    (println "Training :" avg-train-count)
    (println "Testing  :" avg-test-count)
    (println "---------------------")
    (println "Avg RMSE : " avg-rmse)))

;data-split  (split-dataset d 0.7)
;training  (first data-split)
;testing  (second data-split)
         
