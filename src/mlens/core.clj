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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Spark usage
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def c (->
        (conf/spark-conf)
        (conf/master "local[*]")
        (conf/app-name "mlens")))

(def sc (f/spark-context c))

(def data (f/text-file sc "resources/u.data"))

(f/defsparkfn split-line 
  [line]
  (clojure.string/split line #"\t"))

(f/defsparkfn parse-rating 
  [[user-id product-id rating timestamp]]
  (Rating.
    (Integer/parseInt user-id)
    (Integer/parseInt product-id)
    (Double/parseDouble rating)))

(f/defsparkfn user-product-tuple
   [rating]
   (ft/tuple 
     (.user rating) 
     (.product rating)))

(f/defsparkfn wrap-rating
    [r] 
    (ft/tuple 
      (ft/tuple (.user r) (.product r))
      (.rating r)))

(f/defsparkfn unscala
  [r]
  (let  [user-product  (._1 r)
         user  (._1 user-product)
         product  (._2 user-product)
         actual-predicted  (._2 r)
         actual  (._1 actual-predicted)
         predicted  (._2 actual-predicted)]
    [[user product] [actual predicted]]))
;(f/defsparkfn [])
(f/defsparkfn squared-error
    [r]
    (let [r-vals  (last r)]
      (* (- (first r-vals) (second r-vals))
         (- (first r-vals) (second r-vals)))))

(def ratings (-> data
                 (f/map split-line)
                 (f/map parse-rating)))

(def rank 10)
(def numIterations 20)
(def model 
  (ALS/train (.rdd ratings) rank numIterations 0.01))

(def user-products (-> ratings
                       (f/map user-product-tuple)))

(def predictions (-> (.toJavaRDD (.predict model (.rdd user-products)))
                     (f/map-to-pair wrap-rating)))

(def rates (-> ratings
               (f/map-to-pair wrap-rating)))

(def rates-and-predictions (-> (f/join rates predictions)
                               (f/map unscala)))

(def MSE (-> rates-and-predictions
             (f/map squared-error)
             (f/reduce (f/fn [x y] (+ x y)))))


(def user-one (-> rates
                  (f/filter (f/fn [tuple] (= (._1 (._1 tuple)) 1)))
                  (f/map (f/fn [tuple] 
                           [(._2 (._1 tuple)) (._2 tuple)]))
                  (f/filter (f/fn [t] (= (last t) 5.0)))
                  f/collect
                  ))

(def products (-> rates
                  (f/map (f/fn [t] (._2 (._1 t))))
                  f/distinct))

(def one-products (-> products
                      (f/map-to-pair (f/fn [prod-id] (ft/tuple 1 prod-id)))))

(def one-recs (-> (.toJavaRDD (.predict model (.rdd user-products)))
                  f/sort-by-key
                  f/collect))

; Joining two datasets
; (def a-rdd (f/parallelize sc [[1 2] [3 4]]))
; (def b-rdd (f/parallelize sc [[5 6] [7 8]]))
; (f/collect (.union a-rdd b-rdd))

;(clojure.pprint/pprint (.recommendProducts model 2 10))
(def count-ratings (f/count ratings))
(def MSE (/ MSE count-ratings))

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
