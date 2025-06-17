(defpackage :nnl.math.autodiff
  (:use :cl)
  (:shadow + - *) 
  (:export :tensor :make-tensor :repr :repr! :backprop :+ :- :* :matv :matmul :gradient :item))
  