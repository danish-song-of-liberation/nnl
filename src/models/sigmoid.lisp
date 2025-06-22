(in-package :nnl.nn)

(defclass intern-sigmoid () ()
  (:documentation "todo"))
  
(defmethod forward ((nn intern-sigmoid) input-data)
  (nnl.math:.sigmoid input-data))
  
(defmethod parameters ((nn intern-sigmoid))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  