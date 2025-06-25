(defpackage :nnl.math.autodiff
  (:use :cl)
  (:shadow + - * step) 
  (:export :tensor :make-tensor :repr :repr! :backprop :+ :- :* :matv :matmul :activation :mse :zero-grad-once! :zero-grad! :step :step! :gradient :item :mae
   :axpy :axpy! :hstack :vstack))
  