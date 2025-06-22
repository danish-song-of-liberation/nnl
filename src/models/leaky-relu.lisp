(in-package :nnl.nn)

(defclass intern-leaky-relu () ()
  (:documentation "todo"))
  
(defmethod forward ((nn intern-leaky-relu) input-data)
  (nnl.math:.leaky-relu input-data))
  
(defmethod parameters ((nn intern-leaky-relu))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  