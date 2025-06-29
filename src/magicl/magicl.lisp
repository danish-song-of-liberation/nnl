(defpackage :nnl.magicl
  (:use :cl)
  (:shadow abs)
  (:export :coerce-to-tensor :make-random-data :get-magicl-type :get-magicl-type! :transpose :outer :with-broadcast :sum
   :copy-tensor :numerical :abs! :abs :zeros-like :slice :trew))
  