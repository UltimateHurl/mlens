(ns mlens.core
  (:require [flambo.api :as f]
            [flambo.conf :as conf])
  (:import [org.apache.log4j Level Logger]
           [scala Tuple2]
           [org.apache.spark.api.java.function Function]
           [org.apache.spark.mllib.recommendation ALS MatrixFactorizationModel Rating])) 

;; We don't need to see everything;
(.setLevel (Logger/getRootLogger) Level/WARN)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Spark usage
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def c (->
        (conf/spark-conf)
        (conf/master "local[*]")
        (conf/app-name "mlens")))

(def sc (f/spark-context c))

(def data (f/text-file sc "resources/u.data"))

(defn parse-rating 
  [[user-id product-id rating timestamp]]
  (Rating.
    (Integer/parseInt user-id)
    (Integer/parseInt product-id)
    (Double/parseDouble rating)))

(def ratings (-> data
                 (f/map (f/fn [line] (clojure.string/split line #"\t")))
                 (f/map (f/fn [rating-tuple] (parse-rating rating-tuple)))))

(def rank 10)
(def numIterations 20)
(def model 
  (ALS/train (.rdd ratings) rank numIterations 0.01))


(def user-products (-> ratings
                       (f/map (f/fn [rating] [(.user rating) 
                                              (.product rating)]))
                       (f/map (f/fn [r] (Tuple2. (first r) (last r))))))  ;convert to scala Tuple

(def predictions (-> (.toJavaRDD (.predict model (.rdd user-products)))
                     (f/map-to-pair (f/fn [r] 
                                      (Tuple2. 
                                        (Tuple2. (.user r) (.product r))
                                        (.rating r))))))

(def rates (-> ratings
               (f/map-to-pair (f/fn [r] 
                                (Tuple2. 
                                  (Tuple2. (.user r) (.product r))
                                  (.rating r))))))

(def rates-and-predictions (f/join rates predictions)) 

(def MSE (-> rates-and-predictions
             (f/map (f/fn [r] (let [r-vals (._2 r)]
                                (- (._1 r-vals) (._2 r-vals)))))
             (f/reduce (f/fn [x y] (+ x y)))))

(def count-rec  (f/count rates-and-predictions))
(def MSE  (/ MSE count-rec))

(def RMSE (Math/sqrt MSE))

(defn evaluate-predictions 
  [ratings predictions]
  (let [rates-and-predictions (f/join rates predictions)
        SSE (-> rates-and-predictions
                (f/map (f/fn [r] (let [r-vals (._2 r)]
                                  (* (- (._1 r-vals) (._2 r-vals))
                                     (- (._1 r-vals) (._2 r-vals))))))
                (f/reduce (f/fn [x y] (+ x y))))
        count-rec  (f/count rates-and-predictions)
        MSE  (/ SSE count-rec)
        RMSE (Math/sqrt MSE)
        ]
    
    (println "---------------------")
    (println "MSE:    " MSE)
    (println "RMSE:   " RMSE)
    (println "---------------------")))
