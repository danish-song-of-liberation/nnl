#| Philipp Mainlender (1841 - 1876)
   -
   Gott ist gestorben und sein 
   tod war das leben der welt. |#

(asdf:defsystem :nnl
  :license "MIT"
  :version "0.1.0"
  :homepage "https://github.com/danish-song-of-liberation/nnl"
  :description "Common lisp neural networks mvp framework"
  :depends-on (:fiveam :magicl)
  :serial nil
  :components ((:module "src"
				:components ((:file "config" :type "lisp")
				
							 (:module "magicl"
							  :components ((:file "magicl" :type "lisp")
										   (:file "coerce-data" :type "lisp")	
										   (:file "operations" :type "lisp")
										   (:file "deep-copy" :type "lisp")
										   (:file "random" :type "lisp")
										   (:file "broadcasting" :type "lisp")
									       (:file "high-level" :type "lisp")
										   (:file "numgrad" :type "lisp")))
										   
							 (:module "math"
							  :components ((:file "math" :type "lisp")
										   (:file "phi" :type "lisp")
										   
										   (:module "autodiff"
											:components ((:file "autodiff" :type "lisp")
														 (:file "autodiff-d" :type "lisp")
														 (:file "autodiff-i" :type "lisp")
														 (:file "autodiff-h" :type "lisp")))
														 
											(:file "basic-accessors" :type "lisp")
											(:file "broadcasting" :type "lisp")
											(:file "losses" :type "lisp")
											(:file "concat" :type "lisp")
											(:file "tensor-accessors" :type "lisp")))
											
							 (:module "init"
							  :components ((:file "init" :type "lisp")
										   (:file "random" :type "lisp")
										   (:file "xavier-normal" :type "lisp")
										   (:file "xavier-uniform" :type "lisp")))	
											
							 (:module "models"
							  :components ((:file "models" :type "lisp")
										   (:file "fc" :type "lisp")
										   (:file "relu" :type "lisp")
										   (:file "leaky-relu" :type "lisp")
										   (:file "sigmoid" :type "lisp")
										   (:file "tanh" :type "lisp")
										   (:file "activation" :type "lisp")
										   (:file "mlp" :type "lisp")
										   (:file "_unirnn-internal-implementation" :type "lisp")
										   (:file "_birnn-internal-implementation" :type "lisp")
										   (:file "vanilla-rnn" :type "lisp")
										   (:file "sequential" :type "lisp")))
														 
							 (:module "optims"
							  :components ((:file "optims" :type "lisp")
										   (:file "gd" :type "lisp")
										   (:file "momentum" :type "lisp")))
											
							 (:module "utils"
							  :components ((:file "utils" :type "lisp")
										   (:file "clip-grad" :type "lisp")))
										   
							 (:module "api"
							  :components ((:file "api" :type "lisp")
										   (:file "quick" :type "lisp")
										   (:file "highlevel" :type "lisp")))))
					
			   (:module "tests"
			    :components ((:module "math"
							  :components ((:file "math-tests" :type "lisp")
										   (:file "phi-tests" :type "lisp")))
										   
							 (:module "init"
							  :components ((:file "init-tests" :type "lisp")
										   (:file "random-tests" :type "lisp")
										   (:file "xavier-normal-test" :type "lisp")
										   (:file "xavier-uniform-test" :type "lisp")))
										   
							 (:file "tests"))))
							 
	:perform (asdf:test-op (o c)
               (uiop:symbol-call :nnl.tests :run-all-tests)))
						 																						
						
				
						