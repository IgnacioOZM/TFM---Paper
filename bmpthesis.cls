% -----------------------------------------------------------------------------------
% ------------------------------- DOCUMENT PROPERTIES -------------------------------
% -----------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e} 
\ProvidesClass{bmpthesis}[2019/08/19 v1.0.0 Class to write Bachelor, Master, and PhD theses at Comillas Pontifical University]
                                    
% -----------------------------------------------------------------------------------
% -------------------------------- LaTeX EXTENSIONS ---------------------------------
% -----------------------------------------------------------------------------------                                    

\RequirePackage{calc}        % Mathematical operations within LaTeX commands
\RequirePackage{etex}        % Extended TeX engine
\RequirePackage{etoolbox}    % Tools for class and package authors
\RequirePackage{expl3}       % LaTeX3 functions
\RequirePackage{iftex}       % Allows to check the TeX engine used
\RequirePackage{iflang}      % Allows to check the language selected                                                          
\RequirePackage{forloop}     % Allows to use for loops
\RequirePackage{xifthen}     % Allows conditionals in LaTeX format
                             % Extends ifthen package (requires etex)
\RequirePackage{xstring}     % Perform operations on strings

% Fix some known LaTeX issues for pre-2015 kernel versions
% http://tex.stackexchange.com/questions/287146/how-to-use-fixltx2e-only-when-necessary
% Allows subindexes in text mode (\textsubscript{})
\begingroup\expandafter\expandafter\expandafter\endgroup
\expandafter\ifx\csname IncludeInRelease\endcsname\relax
  \RequirePackage{fixltx2e}
\fi

% -----------------------------------------------------------------------------------
% -------------------------------- CLASS PROPERTIES ---------------------------------
% -----------------------------------------------------------------------------------                                

\RequirePackage{kvoptions}   % Key-value class and package options
  %\SetupKeyvalOptions{      % Modify family name and prefix
  %  family = bmpthesis,     % Defaults to class or package name if not set
  %  prefix = bmpthesis@
  %}

% Pass unknown options to the base class
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{book}}

\ProcessKeyvalOptions*  % Process given options
\LoadClass{book}     % Load base class    
                                        
% -----------------------------------------------------------------------------------
% -------------------------------------- FONTS --------------------------------------
% -----------------------------------------------------------------------------------

% Encoding
\RequirePackage[T1]{fontenc}     % Output: Works for most European languages
\RequirePackage[utf8]{inputenc}  % Input: Allows to type special characters using
                                 % regular keyboard commands. Options: latin1, utf8

\RequirePackage{microtype}  % Enhances presentation subtly (justification...), 
                            % especially in multicolumn environments. If it throws
                            % an error try [protrusion=true, expansion=true]
							                       
% Fonts
\RequirePackage{charter}                  % Serif: Bitstream Charter w/o math support
\RequirePackage[defaultsans]{opensans}    % Sans: Open Sans
\RequirePackage[varqu,varl]{inconsolata}  % Monospaced: Inconsolata
\RequirePackage{amsfonts}                 % American Mathematical Society fonts
\RequirePackage{upgreek}                  % Upright greek letters

% Use Ubuntu font for titling if installed; otherwise use Open Sans
\IfFileExists{ubuntu.sty}{
  \RequirePackage[none]{ubuntu}
  \providecommand{\titlingfont}{\fontUbuntuMedium}
}{
  \providecommand{\titlingfont}{\fontseries{sb}\fosfamily} 
}

% Other serif fonts
%\RequirePackage{fourier}              % Utopia text with Fourier math
%\RequirePackage{kerkis}               % Extended Bookman
%\RequirePackage{libertine}            % Linux Libertine
%\RequirePackage{pslatex}              % Times New Roman w/ math support
%\RequirePackage{times}                % Times New Roman w/o math support
%\RequirePackage{tgtermes}             % Extended Nimbus Roman
%\RequirePackage[bitstream-charter]{mathdesign}  % Bitstream Charter w/ math support
  %\let\circledS\undefined                       % Prevent incompatibility with amssymb

% Other sans fonts
%\RequirePackage[scaled]{helvet}          % Helvetica scaled (0.95) to match Times New Roman
%\RequirePackage[defaultsans]{cantarell}  % Cantarell
%\RequirePackage[defaultsans]{droidsans}  % Droid Sans

% Monospaced fonts
%\RequirePackage{courier}                 % Courier
%\renewcommand*\ttdefault{lmtt}           % Latin Modern Typewriter

% Fonts accessible via \fontfamily 
% (e.g. \renewcommand\rmdefault{bch} or \fontfamily{bch}\selectfont)
% pag                    Avant Garde
% fve                    Bera Serif
% fvs                    Bitstream Vera Sans
% pbk                    Bookman
% bch                    Charter
% ccr                    Computer Concrete
% cmr                    Computer Modern
% cmtt                   Computer Modern Typewriter
% pcr                    Courier
% mdugm                  Garamond
% phv                    Helvetica
% fi4                    Inconsolata
% jkp                    Kp-font Roman
% jkpss                  Kp-font Sans
% lmr                    Latin Modern
% lmss                   Latin Modern Sans
% lmtt                   Latin Modern Typewriter
% lmvtt                  Latin Modern Typrewriter Proportional
% LinuxBiolinumT-OsF     Linux Biolinum (formerly 'fxb')
% LinuxLibertineT-OsF    Linux Libertine (formerly 'fxl')
% pnc                    New Century Schoolbook
% ppl                    Palatino
% pplx                   Palatino w/ math support (Mathpazo)
% ptm                    Times
% uncl                   Uncial
% futs                   Utopia (also 'put')
% pzc                    Zapf Chancery

% Symbols
\RequirePackage{bbding}
  \let\Cross\relax            % Prevent clash with marvosym due to redefinition of \Cross
\RequirePackage{dictsym}
\RequirePackage{fontawesome}
\RequirePackage{marvosym} 
\RequirePackage{eurosym}      % Must be loaded after marvosym
\RequirePackage{pifont}

% -----------------------------------------------------------------------------------
% ------------------------------------ LANGUAGE -------------------------------------
% -----------------------------------------------------------------------------------

\RequirePackage{csquotes}                 % Smart quotes. Required by babel. 
                                          % Must go after inputenc
\RequirePackage{babel}                    % Language must be passed as a class option
%\RequirePackage[spanish,english]{babel}  % The last language listed is the default
  %\decimalpoint                          % Use decimal point in those languages in
                                          % which it is not the default symbol
  \hyphenation{hy-phen-ation}             % Correct hyphenation for selected words
  
  % English names
  \addto\captionsenglish{%
    \def\authorname{Author}
    \def\bibname{References}%
    \def\ofname{of}%
    \def\supervisorname{Supervisor}%
    \def\supervisorsname{Supervisors}%
  }  
  
  % Spanish names
  \addto\captionsspanish{%
    \def\authorname{Autor}%
    \def\contentsname{Índice general}%
    \def\listtablename{Índice de tablas}%
    \def\supervisorname{Director}%
    \def\supervisorsname{Directores}%
    \def\tablename{Tabla}%
  }
  
% -----------------------------------------------------------------------------------
% ----------------------------- USER DEFINED PARAMETERS -----------------------------
% -----------------------------------------------------------------------------------                                    

\providecommand{\theAuthor}{}
\providecommand{\theDate}{}
\providecommand{\theDegree}{}
\providecommand{\theKeywords}{}
\providecommand{\theSubject}{}
\providecommand{\theSupervisors}{}
\providecommand{\theTitle}{}
\providecommand{\theFooter}{}

\providecommand{\Author}[1]{\renewcommand{\theAuthor}{#1}}
\providecommand{\Date}[1]{\renewcommand{\theDate}{#1}}
\providecommand{\Degree}[1]{\renewcommand{\theDegree}{#1}}
\providecommand{\Keywords}[1]{\renewcommand{\theKeywords}{#1}}
\providecommand{\Subject}[1]{\renewcommand{\theSubject}{#1}}
\providecommand{\Supervisors}[1]{\renewcommand{\theSupervisors}{#1}}
\providecommand{\Title}[1]{\renewcommand{\theTitle}{#1}}
\providecommand{\Footer}[1]{\renewcommand{\theFooter}{#1}}

% -----------------------------------------------------------------------------------
% ----------------------------------- PAGE LAYOUT -----------------------------------
% -----------------------------------------------------------------------------------       

% Margins
% Use option showframe to display the margins
\RequirePackage[left=3cm, right=2.25cm, top=2.25cm, bottom=2.25cm]{geometry}
\RequirePackage{changepage}  % Allows different margins in selected pages

% Line spacing
\RequirePackage{setspace}    % Keeps single spacing in footnotes and similar environments
  \setstretch{1.10}          % Other options: \singlespacing, \onehalfspacing,
                             % and \doublespacing
                             
% Paragraphs    
\RequirePackage[parfill]{parskip}  % Space between paragraphs. Requires to manually
                                   % correct all extra space introduced in headings
\setlength{\parindent}{15pt}       % Indentation. Must go after parskip
		
% Columns and rows (both in tables and text)
\RequirePackage{multicol}
  \setlength{\columnsep}{1cm}      % Column separation
  \setlength{\columnseprule}{1pt}  % Draw a vertical line to separate columns
\RequirePackage{multirow}
\RequirePackage{vwcol}             % Variable width columns
		
% Avoid hyphenation when possible (-10000 encourage, 10000 forbid)
\hyphenpenalty       =  7500
\tolerance           =  1000
  
% Avoid widow and club lines when possible (-10000 encourage, 10000 forbid)
\widowpenalty        = 10000
\displaywidowpenalty = 10000
\clubpenalty         = 10000
	  
% Ignore layout errors below these thresholds
%\hbadness 1414  % Horizontal badness
%\hfuzz 0.3pt    % Horizontal overfull boxes
%\vfuzz 0.3pt    % Vertical overfull boxes

% ----------------------------------------------------------------------------------- 
% ------------------------------------- COLORS --------------------------------------
% ----------------------------------------------------------------------------------- 

% Use cmyk option for printing could be advisable
\RequirePackage[dvipsnames,hyperref,table]{xcolor}

  % User defined colors
  % \definecolor can be used instead but it overwrites previous color names
  \providecolor{yellowComillas}{RGB}{225,184,28}
  \providecolor{grayComillas}{RGB}{83,86,90}
  \providecolor{blueComillasIcai}{RGB}{0,53,148}
  \providecolor{redComillasIcade}{RGB}{164,18,63}
  \providecolor{blueComillasCihs}{RGB}{123,164,219}
  \providecolor{darkblue}{RGB}{0,67,134}  
  \providecolor{darkgreen}{RGB}{0,145,92}
  \providecolor{darkred}{rgb}{.647,.129,.149}
  \providecolor{subtlegray}{gray}{0.9}
  \providecolor{greenC++}{rgb}{0,0.5,0}
  \providecolor{redC++}{rgb}{0.64,0.08,0.08}
  \providecolor{grayPython}{gray}{0.4}
  \providecolor{greenPython}{rgb}{0,0.5,0}
  \providecolor{orangePython}{HTML}{CC7832}
  \providecolor{violetPython}{rgb}{0.75,0.12,0.95}
  \providecolor{greenMatlab}{rgb}{0.13,0.54,0.13}
  \providecolor{violetMatlab}{rgb}{0.75,0.12,0.95}
  \providecolor{white}{rgb}{1,1,1}

% ----------------------------------------------------------------------------------- 
% HEADINGS
% ----------------------------------------------------------------------------------- 

% Titles
%\RequirePackage{sectsty}
%  \allsectionsfont{\fontfamily{lmss}\selectfont}

% Section numbering
\setcounter{secnumdepth}{5}  % Last level numbered (max. 5): subparagraph

	% Letras capitales
		\RequirePackage{lettrine}
			\setlength{\DefaultFindent}{0.2em}	 % Block text line - letter separation
			\setlength{\DefaultNindent}{0em}		 % Text block- letter separation
			\setcounter{DefaultLines}{3}				 % Number of lines used by the capital letter

\RequirePackage{epigraph}
 \setlength{\epigraphwidth}{0.4\textwidth}   % Total width
  \renewcommand{\epigraphflush}{flushright}  % Block alignment
  \renewcommand{\textflush}{flushright}      % Quote alignment
  \renewcommand{\sourceflush}{flushright}    % Author alignment
  \renewcommand{\epigraphsize}{\normalsize}  % Font size
  \setlength{\epigraphrule}{0pt}
  \setlength{\beforeepigraphskip}{-10pt}
  %\setlength{\afterepigraphskip}{0pt}
		
	% Marcas de agua	
		\usepackage{watermark}
		
% Part page format
\RequirePackage[newparttoc]{titlesec}

% Part format for Thesis with larger fonts (not for PFC where documents must be this size)
%\titleformat{\part}[display]
%	{\vspace*{\fill}\fontsize{40}{44}\fontfamily{ppl}\bfseries\selectfont}
%	{\filcenter\fontsize{36}{32}\selectfont\MakeUppercase{\partname}\hspace{10pt}\thepart\\
%	\hspace{7pt}\rule{75pt}{2.5pt}\vspace{15pt}
%	\thiswatermark{\centering\put(268,-802){\includegraphics[scale=1.5]{ComillasWatermark.pdf}}}}
%	{0mm}
%	{\filcenter\MakeUppercase}
%	[\vspace*{\fill}\thispagestyle{empty}]

  \titleformat{\part}[display]
	{\vspace*{\fill}\fontsize{36}{40}\fontfamily{ppl}\bfseries\selectfont}
	{\filcenter\fontsize{30}{32}\selectfont\MakeUppercase{\partname}\hspace{8pt}\thepart\\
	\hspace{7pt}\rule{75pt}{2.5pt}\vspace{15pt}
	\thiswatermark{\centering\put(268,-802){\includegraphics[scale=1.5]{ComillasWatermark.pdf}}}}
	{0mm}
	{\filcenter\MakeUppercase}
	[\vspace*{\fill}\thispagestyle{empty}]

    \titleformat{\chapter}[display]{\raggedleft\Huge}
    %{\color{gray}\fontfamily{put}\fontsize{36}{0}\selectfont\chaptertitlename\hspace{15pt}\fontfamily{pbk}\fontsize{110}{0}\selectfont\thechapter
    %{\color{gray}\fontfamily{pbk}\fontsize{110}{0}\selectfont\thechapter
    %{\color{gray}\fontfamily{pbk}\fontsize{100}{0}\selectfont\thechapter     % WEB
    {\color[gray]{0.65}\fontfamily{pbk}\fontsize{100}{0}\selectfont\thechapter % PRINT
    %\vspace{0.1cm}
    %\begin{rotate}{90}
    %\hspace{0.4cm}
    %\fontsize{25pt}{0pt}\selectfont \color[gray]{0.75}\chaptertitlename
    %\end{rotate}
    }
    {10pt}
    %{\fontfamily{put}\fontsize{40}{40}\selectfont}
    {\fontfamily{put}\fontsize{30.7}{30.7}\selectfont}
\titlespacing*{\chapter}{0pt}{50pt}{30pt plus 5pt minus 5pt}		
	
%\titleformat{\chapter}[display]
% {\raggedleft}% format applied to label+text
% {\resizebox{!}{2.0cm}{\colorbox{nicered}{\color{white}\bfseries\thechapter}}}% label
% {-2\baselineskip}% separation between label and title body
% {\vspace{2\baselineskip}\vspace{1ex}\Huge\bfseries\color{nicered}}% before the title body
%%[\vspace{1ex}]% after the title body 
%\titlespacing*{\chapter}{0pt}{50pt}{20pt plus 5pt minus 5pt}	
			
%\titleformat{\chapter}[display]{\huge\bfseries}{\chaptertitlename~\thechapter}{0pt}{\Huge} 
%\titlespacing*{\chapter}{0pt}{50pt}{40pt plus 5pt minus 5pt}																								
\titleformat{\section}{\LARGE\bfseries}{\thesection .}{0.5em}{}
\titlespacing*{\section}{0pt}{1.5ex plus .5ex minus .5ex}{0ex plus .5ex minus 1.25ex}
				
\titleformat{\subsection}{\Large\bfseries}{\thesubsection .}{0.5em}{}
\titlespacing*{\subsection}{0pt}{1.25ex plus .5ex minus .5ex}{0ex plus .5ex minus 1ex}
				
\titleformat{\subsubsection}{\large\bfseries}{\thesubsubsection .}{0.5em}{}
\titlespacing*{\subsubsection}{0pt}{1ex plus .5ex minus .5ex}{0ex plus .25ex minus .25ex}
				
\titleformat{\paragraph}{\normalsize\bfseries}{\theparagraph .}{0.5em}{}
\titlespacing*{\paragraph}{0pt}{.75ex plus .25ex minus .25ex}{0ex plus .25ex minus .25ex}
																
\titleformat{\subparagraph}{\normalsize\bfseries}{\thesubparagraph .}{0.5em}{}
\titlespacing*{\subparagraph}{0pt}{.5ex plus .25ex minus .25ex}{0ex plus .25ex minus .25ex}
				
% ----------------------------------------------------------------------------------- 
% TABLES OF CONTENT
% -----------------------------------------------------------------------------------

% Last level displayed in ToC (max. 5). Current selection: \subsection
\setcounter{tocdepth}{3}     

% Don't include ToC, LoF and LoT in the table of contents, but display bibliography
\RequirePackage[nottoc,notlof,notlot]{tocbibind}

% Format ToC, LoF and LoT
%\RequirePackage{titletoc}   % Alternative package				
\RequirePackage[titles]{tocloft}
%\RequirePackage{tocloft}

%\setlength{\cftaftertoctitleskip}{30pt plus 5pt minus 5pt}
%\setlength{\cftafterloftitleskip}{30pt plus 5pt minus 5pt}
%\setlength{\cftafterlottitleskip}{30pt plus 5pt minus 5pt}

  % Vertical spacing
  %\setlength{\cftbeforetoctitleskip}{50pt}
  %\setlength{\cftbeforeloftitleskip}{63.5pt}
  %\setlength{\cftbeforelottitleskip}{63.5pt}
  \setlength{\cftbeforepartskip}{12pt minus 2pt}
  \setlength{\cftbeforechapskip}{6pt minus 1pt}
  \setlength{\cftbeforesecskip}{2pt minus 2pt}
  \setlength{\cftbeforesubsecskip}{2pt minus 2pt}
  \setlength{\cftbeforefigskip}{2pt minus 2pt}
  \setlength{\cftbeforetabskip}{2pt minus 2pt}

  % Text that preceeds the level number
  \renewcommand{\cftpartpresnum}{\partname~}
  %\renewcommand{\cftchappresnum}{\chaptername~}
  \renewcommand{\cftfigpresnum}{\figurename\hfill}
  \renewcommand{\cfttabpresnum}{\tablename\hfill}
 
  % Text that follows the level number
  \renewcommand{\cftpartaftersnum}{.}
  \renewcommand{\cftchapaftersnum}{.}
  \renewcommand{\cftsecaftersnum}{.}
  \renewcommand{\cftsubsecaftersnum}{.}
  \renewcommand{\cftsubsubsecaftersnum}{.}
  \renewcommand{\cftparaaftersnum}{.}
  \renewcommand{\cftsubparaaftersnum}{.}
  \renewcommand{\cftfigaftersnum}{.\hspace{4pt}}
  \renewcommand{\cfttabaftersnum}{.\hspace{4pt}}

  % Leading dots separation
  %\renewcommand{\cftpartdotsep}{3.5}
  %\renewcommand{\cftchapdotsep}{3.5}
  %\renewcommand{\cftsecdotsep}{3.5}
  %\renewcommand{\cftfigdotsep}{0.5}
  %\renewcommand{\cfttabdotsep}{0.5}

%\cftsetpnumwidth{1.25em}                   % Space for the page number
%\cftsetrmarg{1.4em}                        % Distancia del texto al margen derecho (mayor que la anterior)
%\renewcommand{\cftpartnumwidth}{4.4em}     % Espacio para el número de parte
%\renewcommand{\cftfignumwidth}{5.0em}      % Espacio para el número de figura
%\renewcommand{\cfttabnumwidth}{3.5em}      % Espacio para el número de tabla


\settowidth{\cftpartnumwidth}{\cftpartfont\cftpartpresnum III\cftpartaftersnum~}
\settowidth{\cftchapnumwidth}{\cftchapfont\cftchappresnum 8\cftchapaftersnum\hspace{4pt}}
\settowidth{\cftsecnumwidth}{\cftsecfont\cftsecpresnum 8.8\cftsecaftersnum\hspace{4pt}}
\settowidth{\cftsubsecnumwidth}{\cftsubsecfont\cftsubsecpresnum 8.8.8\cftsubsecaftersnum\hspace{4pt}}
\settowidth{\cftsubsubsecnumwidth}{\cftsubsubsecfont\cftsubsubsecpresnum 8.8.8.18\cftsubsubsecaftersnum\hspace{4pt}}
\settowidth{\cftfignumwidth}{\cftfigfont\cftfigpresnum~ 8.88.}
\settowidth{\cfttabnumwidth}{\cfttabfont\cfttabpresnum~ 8.8.}

% Appendix
\ifthenelse{\equal{\cftchappresnum}{\empty}}{}{
  \g@addto@mny\appendix{
    \addtocontents{toc}{
      \unexpanded{\unexpanded{
      \renewcommand{\cftchappresnum}{\appendixname~}
      \settowidth{\cftchapnumwidth}{\cftchappresnum A\cftchapaftersnum\hspace{4pt}}
      }}
    }
  }
}

 % Entry indentation 
  \ifthenelse{\equal{\cftchappresnum}{\empty}}{         
    \settowidth{\cftsecindent}{8\cftchapaftersnum\hspace{4pt}} 
  }{
    \setlength{\cftsecindent}{0pt} 
  }
  \setlength{\cftsubsecindent}{\cftsecindent + \cftsecnumwidth}
  \setlength{\cftsubsubsecindent}{\cftsubsecindent + \cftsubsecnumwidth}
  \setlength{\cftfigindent}{0pt}
  \setlength{\cfttabindent}{0pt}

  % Fonts
  %\renewcommand{\cfttoctitlefont}{\fontfamily{put}\fontsize{36}{0}\selectfont}
  %\renewcommand{\cftloftitlefont}{\fontfamily{put}\fontsize{36}{0}\selectfont}
  %\renewcommand{\cftlottitlefont}{\fontfamily{put}\fontsize{36}{0}\selectfont}
  %\renewcommand{\cftpartfont}{\fontfamily{ppl}\normalsize\bfseries\selectfont}
  %\renewcommand{\cftpartpagefont}{\fontfamily{ppl}\normalsize\bfseries\selectfont}
  %\renewcommand{\cftchapfont}{\fontfamily{ppl}\bfseries\selectfont}
  %\renewcommand{\cftchappagefont}{\fontfamily{ppl}\bfseries\selectfont}

  % Correct an issue in tocloft which makes \cftpartpresnum appear twice
  \renewcommand*{\l@part}[2]{
    \@cftdopartfalse
    \ifnum \c@tocdepth >-2\relax
      \if@cfthaschapter
        \@cftdoparttrue
      \fi
      \ifnum \c@tocdepth >\m@ne
        \if@cfthaschapter\else
          \@cftdoparttrue
        \fi
      \fi
    \fi
    \if@cftdopart
      \if@cfthaschapter
        \addpenalty{-\@highpenalty}
      \else
        \addpenalty\@secpenalty
      \fi
      \addvspace{\cftbeforepartskip}
      \begingroup
        {\leftskip \cftpartindent\relax
        \rightskip \@tocrmarg
        \parfillskip -\rightskip
        \parindent \cftpartindent\relax\@afterindenttrue
        \interlinepenalty\@M
        \leavevmode 
        \@tempdima \cftpartnumwidth\relax
        \let\@cftbsnum \cftpartpresnum
        \let\@cftasnum \cftpartaftersnum
        \let\@cftasnumb \cftpartaftersnumb
        \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
        %{\cftpartfont \cftpartpresnum #1}  % This line has been replaced by the following one
        {\cftpartfont #1}
        \cftpartfillnum{#2}}
        \nobreak
        \if@cfthaschapter
          \global\@nobreaktrue
          \everypar{\global\@nobreakfalse\everypar{}}
        \else
          \if@compatibility
            \global\@nobreaktrue
            \everypar{\global\@nobreakfalse\everypar{}}
          \fi
        \fi
      \endgroup
    \fi
  }
  
% Remove vspace in LoF and LoT between entries of different chapters
\makeatletter
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
%                    \addtocontents{lof}{\protect\addvspace{10\p@}}% NEW
%                    \addtocontents{lot}{\protect\addvspace{10\p@}}% NEW
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\makeatother

% -----------------------------------------------------------------------------------
% ------------------------------ FLOATING ENVIRONMENTS ------------------------------
% -----------------------------------------------------------------------------------

\RequirePackage{float}  % Format options for floats
	
% Images
\RequirePackage[encoding,filenameencoding=latin1]{grffile}  % Path
  \graphicspath{{./images/}}

\RequirePackage{graphicx}
\RequirePackage[export]{adjustbox}  % Allows to put a frame around an image

\ifPDFTeX
  \RequirePackage{epstopdf}  % Convert .eps images to .pdf automatically
    \epstopdfsetup{
      update,    % Only regenerate .pdf files if the .eps file is newer
      prepend,   % Put the .eps extension at the beginning of the graphics list
      suffix =   % Give the resulting .pdf file the same name
    }
\else
  \RequirePackage{epsfig}    % Insert .eps figures
\fi

\RequirePackage[labelformat=simple]{subfig}               % Subfigures
  \renewcommand\thesubfigure{(\alph{subfigure})}          % Autoreference format
  \newcommand{\subfigureautorefname}{\figureautorefname}  % (e.g. Figure 1(b))
  
\RequirePackage{wrapfig}     % Wrap text around figures

\RequirePackage{caption}     % Captions
  \captionsetup{
    font          = small, 
    labelfont     = bf, 
    labelsep      = period, 
    justification = justified, 
    textfont      = small
  }
 	
% Penalize isolated floats	
\renewcommand{\topfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.75}
			
% Tables	
\RequirePackage{array}	            % Column size and alignment options

  % Define left, center and right aligned columns with fixed width
  \newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
  \newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
  \newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
  
\RequirePackage{booktabs}           % Table format
%\RequirePackage{hhline}            % Additional line formats
%  \doublerulesepcolor[rgb]{1,1,1}  % Correct color in tables with double lines
                                    % Requires xcolor
\RequirePackage{tabularx}           % Specify the table width. Requires array
\RequirePackage{xtab}               % Allows tables spanning over multiple pages

% Lists
\RequirePackage{enumitem}
  \setitemize{  % Use \setenumerate, \setdescription, or \setlist for other lists
    itemsep   = 0pt, 
    topsep    = 0pt, 
    partopsep = 0pt
  }  
                                                      
  %\renewcommand{\labelitemi}{\tiny$\blacksquare$}  % Change default bullet points
                                                       
% Equations
\interdisplaylinepenalty = 2500  % Allow page breaks for multiline equations 
                                 % (-10000 encourage, 10000 forbid)                                                 

% Rotate floats            
\RequirePackage{rotating}  % Must be loaded after graphicx

% -----------------------------------------------------------------------------------
% ------------------------------------ FOOTNOTES ------------------------------------
% -----------------------------------------------------------------------------------
	
% Format
\RequirePackage[bottom]{footmisc}  % Must be loaded before fancyhdr
  \renewcommand{\footnoterule}{\noindent\rule{2in}{0.4pt}\vspace{5pt}}
  
\RequirePackage{tablefootnote}     % Easily add footnotes in tables.
                                   % Must be loaded after rotating
	
% Prevent footnote splitting (-10000 encourage, 10000 forbid)
\interfootnotelinepenalty = 10000
												
% -----------------------------------------------------------------------------------
% ------------------------------- HEADERS AND FOOTERS -------------------------------
% -----------------------------------------------------------------------------------

% Margins
%\setlength{\voffset}{5pt}        % Space from the top of the page (1in + \voffset)
\setlength{\headheight}{14pt}			% Header height
%\setlength{\headsep}{20pt}       % Separation between the header and the body
%\addtolength{\textheight}{-15pt}	% Footer height (space removed from the body)
%\setlength{\footskip}{12pt}      % Footer height + Distance to the body.
                                  % The difference with the previous parameter should
                                  % be kept more or less constant (55pt) to preserve
                                  % the appearance

% Format
\RequirePackage{fancyhdr}

  % Separating rules
  \renewcommand{\headrulewidth}{0pt} 
  \renewcommand{\footrulewidth}{0.4pt}

  % Regular header and footer
  \pagestyle{fancy}
    \fancyhf{} % Clear current header and footer

    %\renewcommand{\chaptermark}[1]{%
    %  \ifnum\value{chapter}>0
    %     \markboth{\chaptername~\thechapter.~#1}{}
    %   \else
    %     \markboth{\textit{#1}}{}
    %   \fi}

    \appto\appendix{
      \renewcommand{\chaptermark}[1]{\markboth{\appendixname~\thechapter.~#1}{}} % Leftmark
    }

    \renewcommand{\chaptermark}[1]{\markboth{\chaptername~\thechapter.~#1}{}} % Leftmark
    \renewcommand{\sectionmark}[1]{\markright{\thesection.~#1}}               % Rightmark
  
    \fancyhead[LE]{\nouppercase{\fontfamily{put}\selectfont\small\itshape\leftmark}}
    \fancyhead[RO]{\nouppercase{\fontfamily{put}\selectfont\small\itshape\rightmark}}
    \fancyfoot[RO,LE]{\thepage}
    \fancyfoot[LO,RE]{\small \textit{\theFooter}\\ \textbf{\theAuthor}}
		
    % Header and footer for chapter pages, tables of content...
    %\fancypagestyle{frontmatter}{
    %  \fancyhead{} % Clear the header
    %  \fancyhead[RO, LE]{\heads}
    %}
    
    %\appto\frontmatter{\pagestyle{frontmatter}} 
    %\appto\mainmatter{\pagestyle{fancy}} 

    \fancypagestyle{plain}{
      \fancyhead{}            % Clear the header
    }
									 
\RequirePackage{emptypage}	% Blank pages should not have neither header nor footer

% -----------------------------------------------------------------------------------
% ----------------------------------- MATHEMATICS -----------------------------------
% -----------------------------------------------------------------------------------  

% American Mathematical Society macros
\RequirePackage[cmex10]{amsmath}  % cmex10 ensures only type 1 fonts will be used
\RequirePackage{amssymb}          % Symbols
\RequirePackage{amsthm}           % For theorems

\RequirePackage{cancel}           % Cancel equation terms
\RequirePackage{mathtools}        % Starred versions of matrix environments
\RequirePackage{xfrac}            % Fraction formats  
				
% Additional mathematical operators	
\DeclareMathOperator{\sen}{sen}
\DeclareMathOperator{\ctg}{ctg}
\DeclareMathOperator{\arcctg}{arcctg}
\DeclareMathOperator{\atan}{atan}

% Additional mathematical macros
\providecommand{\cspace}{\hspace{1pt}}                         % Space betweeen variables
\providecommand{\Partial}[2]{\frac{\partial #1}{\partial #2}}  % Partial derivative

% -----------------------------------------------------------------------------------
% -------------------------------------- CODE ---------------------------------------
% -----------------------------------------------------------------------------------  

% \RequirePackage{verbatim}
\RequirePackage{listings}
			
% Visual C++
\providecommand{\VisualCpp}{
  \lstset{
    language          = [Visual]C++,            % Language          
    extendedchars     = true,                   % Extended ASCII
    basicstyle        = \ttfamily \scriptsize,  % Default format
    keywordstyle      = \color{blue},           % Keyword format							
    commentstyle      = \color{greenC++},       % Comment format
    stringstyle       = \color{redC++},         % String format
    breaklines        = true,                   % Allow automatic line breaks
    breakatwhitespace = true,	                  % Allow line breaks only at white spaces
    tabsize           = 4,                      % Number of spaces equal to a tab
    showspaces        = false,                  % Show spaces in the code
    showtabs          = false,                  % Show tabs in the code
    showstringspaces  = false,                  % Show spaces among strings
    %backgroundcolor  = \color{white},          % Background color
    %numbers          = left,                   % Line number positioning
    %numberstyle      = \footnotesize,          % Line number format
    %numbersep        = 10pt,                   % Separation between the line number and the code
    %firstnumber      = auto,                   % Display number in the first row
    %stepnumber       = 10,                     % Gap between two numbered lines
    captionpos        = b,                      % Caption position
    mathescape        = true                    % Allow mathematical expressions between $
  }
}
			
% Matlab
\providecommand{\Matlab}{
  \lstset{
    language          = Matlab,                 % Language
    extendedchars     = true,                   % Extended ASCII
    basicstyle        = \ttfamily \scriptsize,  % Default format
    keywordstyle      = \color{black},          % Keyword format						
    commentstyle      = \color{greenMatlab},    % Comment format
    stringstyle       = \color{violetMatlab},   % String format
    breaklines        = true,                   % Allow automatic line breaks
    breakatwhitespace = true,                   % Allow line breaks only at white spaces
    tabsize           = 4,                      % Number of spaces equal to a tab
    showspaces        = false,                  % Show spaces in the code
    showtabs          = false,                  % Show tabs in the code
    showstringspaces  = false,                  % Show spaces among strings
    %backgroundcolor  = \color{white},          % Background color
    %numbers          = left,                   % Line number positioning
    %numberstyle      = \footnotesize,          % Line number format
    %numbersep        = 10pt,                   % Separation between the line number and the code
    %firstnumber      = auto,                   % Display number in the first row
    %stepnumber       = 10,                     % Gap between two numbered lines
    captionpos        = b,                      % Caption position
    mathescape        = true                    % Allow mathematical expressions between $
  }
}

% Algorithms and pseudocode
\RequirePackage{algorithmicx}
\RequirePackage[algochapter,ruled,noline]{algorithm2e}
  \setlength{\algomargin}{0.8em}
  \SetAlgoCaptionSeparator{.}
  \SetAlCapFnt{\small}
  \SetFuncSty{textrm}
  \SetArgSty{textrm}
  \SetDataSty{textit}
  
\makeatletter
\begingroup
  \let\newcounter\@gobble
  \let\setcounter\@gobbletwo
  \globaldefs\@ne
  \let\c@loadepth\@ne
  \newlistof{alg}{loa}{\listalgorithmcfname}
\endgroup
\let\l@algocf\l@alg
\makeatother

%\setlength{\cftbeforeloatitleskip}{63.5pt}
\setlength{\cftbeforealgskip}{2pt minus 2pt}
\renewcommand{\cftalgaftersnum}{.\hspace{4pt}}
\renewcommand{\cftalgpresnum}{Algorithm\hfill}
\settowidth{\cftalgnumwidth}{\cftalgfont\cftalgpresnum~ 8.8.}
\setlength{\cftalgindent}{0pt}

% Remove vspace in LoF and LoT between entries of different chapters
\makeatletter
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    %\addtocontents{lof}{\protect\addvspace{10\p@}}% NEW
                    %\addtocontents{lot}{\protect\addvspace{10\p@}}% 
                    %\addtocontents{loa}{\protect\addvspace{10\p@}}% NEW
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\makeatother

% -----------------------------------------------------------------------------------
% -------------------------------------- UNITS --------------------------------------
% -----------------------------------------------------------------------------------

% Text units
\renewcommand{\textohm}{\ensuremath{\Omega}}

\RequirePackage[tight,nice]{units}   % Simple units
\RequirePackage{siunitx}             % Advanced units
  \sisetup{
    detect-all,                      % Use the current font and weight
    retain-explicit-plus,            % Display leading plus sign if present
    per-mode         = symbol,       % Use a slash to indicate units in the denominator
    range-phrase     = --,           % Use an en-dash for ranges
    output-product   = \cdot,        % Use a centered dot for products
    exponent-product = \cdot         % Use a centered dot for scientific notation
  }
  
  % Spanish language specific options
  \IfStrEq*{\languagename}{spanish}{
    \sisetup{output-decimal-marker = {,}}  % Use decimal comma
  }{}
  
  % Additional units   
  \DeclareSIUnit{\fahrenheit}{\SIUnitSymbolDegree F}
  \DeclareSIUnit{\degreeFahrenheit}{\fahrenheit}
  \DeclareSIUnit{\ntu}{NTU}          % Nephelometric Turbidity Unit
  \DeclareSIUnit{\ohm}{\ensuremath{\Omega}}
  \renewcommand{\SIUnitSymbolMicro}{\ensuremath{\upmu}}

% -----------------------------------------------------------------------------------
% ----------------------------------- MISCELLANY ------------------------------------
% ----------------------------------------------------------------------------------- 

\RequirePackage{comment}             % Comment text out
\RequirePackage{pdfpages}            % Attach .pdf files
\RequirePackage{qrcode}              % Generate QR codes
\RequirePackage[normalem]{ulem}      % Underlining
  
% -----------------------------------------------------------------------------------
% More miscellany
% ----------------------------------------------------------------------------------- 

\RequirePackage{tikz}               % Vector graphics

% Format quote (requires etoolbox)
%\patchcmd{\quote}{\rightmargin}{\leftmargin 4em \rightmargin}{}{}

\usepackage{datetime}
  \newdateformat{mydateformat}{\monthname[\THEMONTH]~\THEYEAR}

% Place text anywhere in the paper
\RequirePackage[absolute]{textpos}
  \setlength{\TPHorizModule}{\paperwidth}
  \setlength{\TPVertModule}{\paperheight - \voffset}

% -----------------------------------------------------------------------------------
% BIBLIOGRAPHY
% -----------------------------------------------------------------------------------

%\RequirePackage[compress,sort]{cite}
%\renewcommand\citeleft{}
%\renewcommand\citeright{}
%\renewcommand\citeform[1]{[#1]}

%\RequirePackage[square,sort&compress,numbers]{natbib}

\RequirePackage[backend=biber,
            %bibstyle=ieee, %ieee-alphabetic
            bibencoding=utf8,
            bibstyle=ieee-alphabetic,
            citestyle=alphabetic,
            %citestyle=numeric-comp,
            natbib=true,
            hyperref=true,
            url=true,
            doi=false,
            isbn=false,
            backref=false,
            block=none, %space
            maxcitenames=2,
            mincitenames=1
            %maxnames=2,
            %maxbibnames=10,
            %minbibnames=10
            ]{biblatex} 
            
\RequirePackage{xpatch}
\xpatchbibdriver{online}
  {\printtext[parens]{\usebibmacro{date}}}
  {\iffieldundef{year}
    {}
    {\printtext[parens]{\usebibmacro{date}}}}
  {}
  {\typeout{There was an error patching biblatex-ieee (specifically, ieee.bbx's @online driver)}}
         
\renewcommand{\bibname}{References}   
\renewcommand{\bibfont}{\normalfont\small}
              

%\renewcommand{\newunitpunct}{\addcomma\space}
%\renewcommand{\intitlepunct}{\space}

%\DefineBibliographyStrings{english}{%
%    backrefpage  = {\lowercase{s}ee p.}, % for single page number
%    backrefpages = {\lowercase{s}ee pp.} % for multiple page numbers
%}

%\addbibresource{References.bib}
     
% -----------------------------------------------------------------------------------
% ENLACES, REFERENCIAS CRUZADAS Y NUEVOS ÃNDICES DE CONTENIDO
% -----------------------------------------------------------------------------------

\RequirePackage{longtable}
\RequirePackage[only-used = false]{acro} % Acronyms \ac{identificador}
%\newlist{acronyms}{description}{1}
%\setlist[acronyms]{labelwidth=3em,leftmargin=3.5em,noitemsep,itemindent=0pt}

% -----------------------------------------------------------------------------------
% -------------------------------- CROSS-REFERENCES ---------------------------------
% -----------------------------------------------------------------------------------

\ifPDFTeX
  \RequirePackage[pdftex]{hyperref}  % Must be loaded at the end
\else
  \RequirePackage{hyperref}
\fi
  \hypersetup{
    pdffitwindow = true,
    pdftitle           = \theTitle,
    pdfauthor          = \theAuthor,
    pdfsubject         = \theSubject,
    pdfkeywords        = \theKeywords,
    pdfnewwindow       = true,
    %colorlinks        = true,  % Cannot be set if menukeys is used
    citecolor          = black,
    filecolor          = black,
    linkcolor          = black,
    urlcolor           = blue,
    breaklinks         = true,
    linktocpage        = false,
    bookmarksnumbered  = true,
    bookmarksopen      = true,
    bookmarksopenlevel = -1
  }
  
% Keyboard like keys. Must be loaded after hyperref because otherwise there
% is a clash with the hyperref's colorlinks. In order to preserve hyperref's
% functionality, the option hyperrefcolorlinks must be set.
\RequirePackage[hyperrefcolorlinks,os=win]{menukeys}   
  \renewmenumacro{\directory}[/]{pathswithfolder}
  
  %\usepackage[hyperpageref]{backref}
  %\renewcommand*{\backref}[1]{(see p.~#1)}
  
\RequirePackage[all]{hypcap}  % Correct hyperlink anchors position in floats.
                              % Must be loaded after hyperref

% Modify automatic reference names
\addto\extrasenglish{%
  \def\partautorefname{Part}%
  \def\chapterautorefname{Chapter}%
  \def\sectionautorefname{Section}%
  \def\subsectionautorefname{Section}%
  \def\subsubsectionautorefname{Section}%
  \def\figureautorefname{Figure}%
  \def\subfigureautorefname{Figure}%
  \def\tableautorefname{Table}%
  \def\equationautorefname{Equation}%
  \def\codeautorefname{Code}%
  \def\algorithmautorefname{Algorithm}%
} 

\addto\extrasspanish{%
  \def\partautorefname{Parte}%
  \def\chapterautorefname{Capítulo}%
  \def\sectionautorefname{Sección}%
  \def\subsectionautorefname{Sección}%
  \def\subsubsectionautorefname{Sección}%
  \def\figureautorefname{Figura}%
  \def\subfigureautorefname{Figura}%
  \def\tableautorefname{Tabla}%
  \def\equationautorefname{Ecuación}%
  \def\codeautorefname{Código}%
  \def\algorithmautorefname{Algoritmo}%
}

% -----------------------------------------------------------------------------------
		
	% Dar formato a los nombres automÃ¡ticos de las referencias
		%\usepackage[capitalise]{cleveref}
		
% Redefinir los niveles de las divisiones del documento para poder incorporar doc en el superior.
	% Para ello hay que aÃ±adir al final de Proyecto Fin de Carrera.out la linea \let\WriteBookmarks\relax
	% despuÃ©s de compilar la versiÃ³n definitiva (Debe ir detrÃ¡s de hyperref)
		\makeatletter
			\renewcommand*{\toclevel@part}{0}
			\renewcommand*{\toclevel@chapter}{1}
			\renewcommand*{\toclevel@section}{2}
			\renewcommand*{\toclevel@subsection}{3}
			\renewcommand*{\toclevel@subsubsection}{4}
			\renewcommand*{\toclevel@paragraph}{5}
			\renewcommand*{\toclevel@subparagraph}{6}
		\makeatother
	
	% Ãndice de extractos de cÃ³digo si se usa el paquete caption (detrÃ¡s de hyperref)
	 	\usepackage{float}
 		\newlistof{codigo}{loc}{Índice de código}
			\newlistentry{code}{loc}{0}
			\newfloat{code}{htbp}{loc}
			\floatname{code}{Código}
		
			%\renewcommand{\cftcodefont}{CÃ³digo }
		 	\renewcommand{\cftcodeaftersnum}{.}
		 	\setlength{\cftcodeindent}{0pt}        % SangrÃ­Â­a del nÃºmero de cÃ³digo
		 	\renewcommand{\cftcodenumwidth}{1.6em} % Espacio para el nÃºmero de cÃ³digo
		 	\renewcommand{\cftafterloctitle}{\thispagestyle{plain}}
		 	\setlength{\cftbeforeloctitleskip}{63.5pt}
		 	%\renewcommand{\cftloctitlefont}{\fontfamily{lmss}\huge\bfseries\selectfont}
		 	
% Enumerate floating environments correlatively throughout the whole document
%\RequirePackage{remreset}  % Remove counters from reset list 
%  \makeatletter
%    \@removefromreset{figure}{chapter}
%	     \renewcommand{\thefigure}{\arabic{figure}}
%	   \@removefromreset{table}{chapter}
%	     \renewcommand{\thetable}{\arabic{table}}
%	   \@removefromreset{equation}{chapter}
%	     \renewcommand{\theequation}{\arabic{equation}}
%    \@removefromreset{footnote}{chapter}
%  \makeatother
		 	
% -----------------------------------------------------------------------------------
% ------------------------------- USER DEFINED MACROS -------------------------------
% -----------------------------------------------------------------------------------                                    

\RequirePackage{macros}
