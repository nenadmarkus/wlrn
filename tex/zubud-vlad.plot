\begin{tikzpicture}
	\begin{axis} [
		xlabel={VLAD size},
		ylabel={NN},
		xmin=512, xmax=8192,
		ymin=0.875, ymax=1,
		legend pos=south east
	]
		\addplot[color=red]
			coordinates {
			(512, 0.98209)(1024, 0.979104)(1536, 0.983085)(2048, 0.98408)(2560, 0.98408)(3072, 0.983085)(3584, 0.985075)(4096, 0.989055)(4608, 0.98607)(5120, 0.985075)(5632, 0.989055)(6144, 0.987065)(6656, 0.98806)(7168, 0.99005)(7680, 0.987065)(8192, 0.99005)
			};
		\addplot[color=green]
			coordinates {
			(512, 0.961194)(1024, 0.980099)(1536, 0.980099)(2048, 0.98607)(2560, 0.983085)(3072, 0.983085)(3584, 0.985075)(4096, 0.98607)(4608, 0.985075)(5120, 0.98408)(5632, 0.989055)(6144, 0.985075)(6656, 0.98806)(7168, 0.98607)(7680, 0.985075)(8192, 0.985075)
			};
		\addplot[color=blue]
			coordinates {
			(512, 0.951244)(1024, 0.963184)(1536, 0.968159)(2048, 0.973134)(2560, 0.978109)(3072, 0.981095)(3584, 0.98209)(4096, 0.977114)(4608, 0.978109)(5120, 0.983085)(5632, 0.98209)(6144, 0.981095)(6656, 0.985075)(7168, 0.983085)(7680, 0.98806)(8192, 0.985075)
			};
		\addplot[color=magenta]
			coordinates {
			(512, 0.9)(1024, 0.95)(2048, 0.962189)(4096, 0.970149)(6144, 0.980099)(8192, 0.983085)(10240, 0.981095)(12288, 0.98408)(14336, 0.985075)(16384, 0.983085)(18432, 0.98408)(20480, 0.985075)(22528, 0.98607)(24576, 0.987065)(26624, 0.98607)(28672, 0.98806)(30720, 0.987065)(32768, 0.985075)
			};
		\addplot[color=cyan]
			coordinates {
			(512, 0.892537)(1024, 0.955224)(1536, 0.966169)(2048, 0.967164)(2560, 0.971144)(3072, 0.973134)(3584, 0.976119)(4096, 0.977114)(4608, 0.977114)(5120, 0.975124)(5632, 0.977114)(6144, 0.98209)(6656, 0.983085)(7168, 0.98408)(7680, 0.98408)(8192, 0.977114)
			};
		%\legend{$32\times32\rightarrow64$,DeepDesc,PN-Net,DeepCompare,SIFT}
	\end{axis}
\end{tikzpicture}
\begin{tikzpicture}
	\begin{axis} [
		xlabel={VLAD size},
		ylabel={FT},
		xmin=512, xmax=8192,
		ymin=0.6, ymax=0.9,
		legend pos=south east
	]
		\addplot[color=red]
			coordinates {
			(512, 0.820149)(1024, 0.854229)(1536, 0.863682)(2048, 0.871642)(2560, 0.871144)(3072, 0.880099)(3584, 0.884577)(4096, 0.884328)(4608, 0.891045)(5120, 0.893781)(5632, 0.889552)(6144, 0.896766)(6656, 0.896269)(7168, 0.892289)(7680, 0.895522)(8192, 0.900497)
			};
		\addplot[color=green]
			coordinates {
			(512, 0.775124)(1024, 0.824627)(1536, 0.84005)(2048, 0.850249)(2560, 0.858209)(3072, 0.862935)(3584, 0.865672)(4096, 0.874129)(4608, 0.874378)(5120, 0.876866)(5632, 0.879353)(6144, 0.876617)(6656, 0.88209)(7168, 0.882587)(7680, 0.883582)(8192, 0.879851)
			};
		\addplot[color=blue]
			coordinates {
			(512, 0.747264)(1024, 0.795771)(1536, 0.814428)(2048, 0.835323)(2560, 0.842786)(3072, 0.849751)(3584, 0.854478)(4096, 0.856716)(4608, 0.854726)(5120, 0.866169)(5632, 0.860199)(6144, 0.863682)(6656, 0.866418)(7168, 0.872637)(7680, 0.872886)(8192, 0.873134)
			};
		\addplot[color=magenta]
			coordinates {
			(512, 0.6)(1024, 0.74)(2048, 0.777114)(4096, 0.81194)(6144, 0.841542)(8192, 0.853731)(10240, 0.869403)(12288, 0.866169)(14336, 0.869652)(16384, 0.878856)(18432, 0.879353)(20480, 0.875622)(22528, 0.881343)(24576, 0.892289)(26624, 0.889801)(28672, 0.89602)(30720, 0.891791)(32768, 0.889304)
			};
		\addplot[color=cyan]
			coordinates {
			(512, 0.638806)(1024, 0.732338)(1536, 0.756468)(2048, 0.775622)(2560, 0.784826)(3072, 0.799254)(3584, 0.801493)(4096, 0.805473)(4608, 0.807463)(5120, 0.812438)(5632, 0.806965)(6144, 0.81592)(6656, 0.817164)(7168, 0.823134)(7680, 0.820647)(8192, 0.817662)
			};
		%\legend{$32\times32\rightarrow64$,DeepDesc,PN-Net,DeepCompare,SIFT}
	\end{axis}
\end{tikzpicture}
\begin{tikzpicture}
	\begin{axis} [
		xlabel={VLAD size},
		ylabel={ST},
		xmin=512, xmax=8192,
		ymin=0.7, ymax=0.95,
		legend pos=south east
	]
		\addplot[color=red]
			coordinates {
			(512, 0.867662)(1024, 0.898507)(1536, 0.906468)(2048, 0.913682)(2560, 0.906219)(3072, 0.920647)(3584, 0.921393)(4096, 0.921891)(4608, 0.923881)(5120, 0.927363)(5632, 0.920647)(6144, 0.931095)(6656, 0.927861)(7168, 0.927363)(7680, 0.927114)(8192, 0.930099)
			};
		\addplot[color=green]
			coordinates {
			(512, 0.83806)(1024, 0.877612)(1536, 0.884577)(2048, 0.895025)(2560, 0.901244)(3072, 0.906219)(3584, 0.905473)(4096, 0.91194)(4608, 0.911443)(5120, 0.914677)(5632, 0.919154)(6144, 0.915174)(6656, 0.914428)(7168, 0.916915)(7680, 0.918657)(8192, 0.918657)
			};
		\addplot[color=blue]
			coordinates {
			(512, 0.805224)(1024, 0.848259)(1536, 0.86393)(2048, 0.880846)(2560, 0.882836)(3072, 0.891294)(3584, 0.890796)(4096, 0.893284)(4608, 0.891294)(5120, 0.900746)(5632, 0.899005)(6144, 0.90199)(6656, 0.904478)(7168, 0.910945)(7680, 0.91393)(8192, 0.910448)
			};
		\addplot[color=magenta]
			coordinates {
			(512, 0.75)(1024, 0.80)(2048, 0.838557)(4096, 0.868159)(6144, 0.889055)(8192, 0.895274)(10240, 0.907711)(12288, 0.901741)(14336, 0.907711)(16384, 0.914179)(18432, 0.915672)(20480, 0.916169)(22528, 0.919652)(24576, 0.927114)(26624, 0.924378)(28672, 0.927363)(30720, 0.929851)(32768, 0.926617)
			};
		\addplot[color=cyan]
			coordinates {
			(512, 0.700498)(1024, 0.79602)(1536, 0.810448)(2048, 0.826368)(2560, 0.838806)(3072, 0.850995)(3584, 0.852985)(4096, 0.850249)(4608, 0.855721)(5120, 0.853731)(5632, 0.852736)(6144, 0.863184)(6656, 0.864677)(7168, 0.86194)(7680, 0.866169)(8192, 0.869901)
			};
		%\legend{$32\times32\rightarrow64$,DeepDesc,PN-Net,DeepCompare,SIFT}
	\end{axis}
\end{tikzpicture}