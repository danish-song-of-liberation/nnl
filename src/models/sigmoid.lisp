(in-package :nnl.nn)

(defclass sigmoid () ()
  (:documentation "todo"))
  
(defmethod forward ((nn sigmoid) input-data)
  (nnl.math:.sigmoid input-data))
  
(defmethod get-parameters ((nn sigmoid))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  
(defmacro sigmoid ()
  `(make-instance 'sigmoid))
    