(defpackage :nnl.optims
  (:use :cl)
  (:shadow step)
  (:export :step :zero-grad :make-optim :gd))

(in-package :nnl.optims)

(defmacro make-optim (name &body keys)
  `(make-instance ,name ,@keys))
  