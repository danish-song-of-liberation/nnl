(defun quick-debug-tensor-value (stream char)
  (declare (ignore char))
  
  (let ((tensor (eval (read stream t nil t))))
    (format t "~a~%" (nnl.math:item tensor))))

(defun quick-debug-execution-time (stream char)
  (declare (ignore char))
  
  (time (eval (read stream t nil t))))
  
(defun quick-debug-tensor-repr (stream char)
  (declare (ignore char))
  
  (nnl.math.autodiff:repr! (eval (read stream t nil t))))	

(set-macro-character #\$ #'quick-debug-tensor-value)
(set-macro-character #\% #'quick-debug-execution-time)
(set-macro-character #\? #'quick-debug-tensor-repr)
