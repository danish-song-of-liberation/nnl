(in-package :nnl.nn)

(defclass intern-leaky-relu () ()
  (:documentation "todo"))
  
(defmethod forward ((nn intern-leaky-relu) input-data &key padding-mask)
  (nnl.math:.leaky-relu input-data))
  
(defmethod get-parameters ((nn intern-leaky-relu))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  
(defmacro leaky-relu ()
  `(make-instance 'intern-leaky-relu))   
  