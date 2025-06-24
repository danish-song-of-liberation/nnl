(defpackage :nnl.math
  (:use :cl)
  (:shadow tanh * + -)
  (:export :relu :leaky-relu :sigmoid :tanh :linear :ufunc :make-tensor :item :gradient :grad :zeros :ones :full :arange :linspace :eye :randn :+ :- :* :matvec :matmul :activation
   :.relu :.leaky-relu :.sigmoid :.tanh :.linear :size :order :shape :at :backprop :with-broadcasting :transpose! :transpose :scale :numerical :instant-mse))
