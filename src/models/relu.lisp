(in-package :nnl.nn)

(defclass intern-relu () ()
  (:documentation "todo"))
  
(defmethod forward ((nn intern-relu) input-data &key padding-mask)
  (nnl.math:.relu input-data))
  
(defmethod get-parameters ((nn intern-relu))
  (list)) ; it should return nil but I'm afraid the optimizer will complain. and in the end, an empty list is the same as nil
  
(defmacro relu ()
  `(make-instance 'intern-relu))  
  