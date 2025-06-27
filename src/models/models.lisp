(defpackage :nnl.nn
  (:use :cl)
  (:shadow tanh)
  (:export :forward :sequential :fc :mlp :relu :leaky-relu :sigmoid :tanh :get-parameters))
  
(in-package :nnl.nn)

(defgeneric forward (model data &key padding-mask))
(defgeneric parameters (model))
  