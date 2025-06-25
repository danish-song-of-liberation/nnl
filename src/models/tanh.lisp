(in-package :nnl.nn)

(defclass tanh () ()
  (:documentation "todo"))
  
(defmethod forward ((nn tanh) input-data)
  (nnl.math:.tanh input-data))
  
(defmethod get-parameters ((nn tanh))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  
(defmacro tanh ()
  `(make-instance 'tanh))   
    