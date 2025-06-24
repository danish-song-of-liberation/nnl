(in-package :nnl.nn)

(defclass intern-tanh () ()
  (:documentation "todo"))
  
(defmethod forward ((nn intern-tanh) input-data)
  (nnl.math:.tanh input-data))
  
(defmethod get-parameters ((nn intern-tanh))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  
(defmacro tanh ()
  `(make-instance 'intern-tanh))   
    