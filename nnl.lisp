(asdf:defsystem :nnl
  :license "MIT"
  :version "0.1.0"
  :depends-on (:fiveam :magicl)
  :components ((:module "src"
				:components ((:module "math"
							  :components ((:module "autodiff"
											:components ((:file "autodiff" :type :lisp-source)
														 (:file "autodiff-d" :type :lisp-source)
														 (:file "autodiff-h" :type :lisp-source)))
											(:file "numgrad" :type :lisp-source)))))))
											