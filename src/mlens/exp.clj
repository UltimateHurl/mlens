(ns mlens.exp)


; The number of ratings each product has
(defn to-product-rating-counts 
  [d] 
  (-> d 
      (f/map-to-pair (f/fn [t] (ft/tuple (second t) 1)))
      (f/reduce-by-key (f/fn [x y] (+ x y)))))

; Ratings a product has as a continous string
(defn to-ratings-as-strings 
  [d]
  (-> d
      (f/map-to-pair (f/fn [t] (ft/tuple (second t) (nth t 2))))
      (f/reduce-by-key (f/fn [x y] (str x y)))))
; misc funcs
(defn sqr  
  [x]  
  (* x x))

(defn variance
  [data]
  (let [mean (double (/ (reduce + data) 
                      (count data)))]
    (/ (reduce + 
               (map #(sqr (- % mean)) data))
       (count data))))

(defn rating-to-map
  [r]
  {:user    (.user r)
   :product (.product r)
   :rating  (.rating r)})

(defn single-point-entropy
  [freq len]
  (let [rf (/ freq len)]
    (Math/abs 
      (* rf (/ (Math/log rf) 
               (Math/log 2))))))

(defn string-entropy
  [s]
  (let  [len  (count s)
         log-2  (Math/log 2)]
    (->> (frequencies s)
         (map  (fn  [[_ v]]
                 (let  [rf  (/ v len)]
                   (->  (Math/log rf)  (/ log-2)  (* rf) Math/abs))))
         (reduce +))))

; Rating risk/reward expected value work
; entropy of counts, points to most popular
(def entropy 
  (into {} (-> product-rating-counts ; to-product-rating-counts
               (f/map (f/fn [t] [(._1 t) 
                                 (single-point-entropy (._2 t) count-ratings)]))
               (f/map (f/fn [t] [(keyword (str (first t)))
                                 (second t)]))
               f/collect)))

; also points to most popular
(def info-gain 
  (into {} (-> product-rating-counts ; to-product-rating-counts
               (f/map (f/fn [t] [(._1 t) 
                                 (- (single-point-entropy (._2 t) count-ratings)
                                    (single-point-entropy (+ 1 (._2 t)) count-ratings))]))
               (f/map (f/fn [t] [(keyword (str (first t)))
                                 (second t)]))
               f/collect)))

; different approach:entropy of string of the ratings
(def entropy-str 
  (into {} (-> product-ratings-as-strings
               (f/map (f/fn [t] [(._1 t) 
                                 (string-entropy (._2 t))]))
               (f/map (f/fn [t] [(keyword (str (first t)))
                                 (second t)]))
               f/collect)))

; normalising by length
(def entropy-str-normalised 
  (into {} (-> product-ratings-as-strings
               (f/map (f/fn [t] [(._1 t)
                                 (/ (string-entropy (._2 t))
                                     (count (._2 t)))]))
               (f/map (f/fn [t] [(keyword (str (first t)))
                                 (second t)]))
               f/collect)))

; just variance rather than entropy
(def variance-map 
  (into {} (-> split-data
               (f/map-to-pair (f/fn [t] (ft/tuple (second t) 
                                                   (Double/parseDouble (nth t 2)))))
               f/group-by-key
               (f/map f/group-untuple)
               (f/map (f/fn [t] (ft/tuple (first t) (variance (second t)))))
               (f/map f/untuple)
               (f/map (f/fn [t] [(keyword (str (first t)))
                                 (second t)]))
               f/collect)))

(defn compare-expected
  [k]
  (clojure.pprint/pprint
    [
    (k entropy)
    (k info-gain)
    (k entropy-str)
    (k entropy-str-normalised)
    (k variance-map)]))

; map entropy to recommendations
(defn lookup-entropy
  [product]
  (let [prod-key (keyword (str product))]
    (prod-key entropy-str)))

(def ent-rate-maps 
  (map #(assoc % :ent-rating (* (:rating %) (:entropy %)))
     (map #(assoc % :entropy (lookup-entropy (:product %)))
          (map rating-to-map (.recommendProducts model 1 10)))))

(def ent-order 
  (map #(:product %) (sort-by :ent-rating ent-rate-maps)))

(def nat-order
  (map #(:product %) (sort-by :rating ent-rate-maps)))

; calculate how each item is moved by entropy
(defn analyse-shift
  [ent-order nat-order]
  (map #(- (first %)
          (second %)) 
      (map vector (range 10) 
                  (map #(.indexOf ent-order %) nat-order))))

; classic q-learning expected value function.
(defn expected-value
  [reward discount q-value]
  (+ reward (* discount q-value)))
