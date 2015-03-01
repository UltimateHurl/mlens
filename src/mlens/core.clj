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
  (map f/untuple (f/tuple r)))

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

(defn experiment
  []
  (let [training (.cache (.sample data false 0.7))
        testing (.subtract data training)
        rank 10
        numIterations 20
        model (ALS/train (.rdd (to-ratings training))
                         rank
                         numIterations
                         0.01)
        predictions (-> (.toJavaRDD 
                          (.predict model 
                                    (.rdd (to-user-products 
                                            (to-ratings testing)))))
                        (f/map-to-pair wrap-rating)) 
        SSE (-> (f/join rates predictions)
                (f/map unscala)
                (f/map squared-error)
                (f/reduce (f/fn [x y] (+ x y))))
        MSE (/ SSE (f/count testing))
        RMSE (Math/sqrt MSE)
        ]
    (println "---------------------")
    (println "Data     : " (f/count data))
    (println "Training : " (f/count training))
    (println "Testing  : " (f/count testing))
    (println "---------------------")
    (println "RMSE     : " RMSE)))

