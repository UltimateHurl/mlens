(defproject mlens "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :main mlens.core
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [yieldbot/flambo "0.5.0-SNAPSHOT"]
                 [org.apache.spark/spark-core_2.10 "1.2.1"]
                 [org.apache.spark/spark-mllib_2.10 "1.2.1"]]
  :checksum :ignore
  :repl-options {:timeout 220000}
  :profiles {:dev
    {:aot [mlens.core]}})
