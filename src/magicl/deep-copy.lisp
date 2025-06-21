(in-package :nnl.magicl)

(defun copy-tensor (tensor)
  (magicl:deep-copy-tensor tensor))
  