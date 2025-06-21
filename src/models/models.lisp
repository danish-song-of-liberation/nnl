(defpackage :nnl.nn
  (:use :cl)
  (:export :forward))
  
(in-package :nnl.nn)

(defgeneric forward (model data))
  