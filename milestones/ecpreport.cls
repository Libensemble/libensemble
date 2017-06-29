%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File  : report/ecpreport.cls
% Author: Seth R Johnson and Tom Evans
% Date  : Thu Dec 08 11:39:45 2016
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ecpreport}[2016/12/08 ECP Report Class]

\ExecuteOptions{a4paper,11pt,oneside,onecolumn,number}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass{article}

\RequirePackage{geometry}
\RequirePackage{fancyhdr}

\RequirePackage{pdfpages}
\RequirePackage{fancyhdr}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage{amsthm}
\RequirePackage{booktabs}
\RequirePackage[hidelinks]{hyperref}
\RequirePackage{xcolor}
\RequirePackage{colortbl}
\RequirePackage{bm} % bold math symbols for vectors
\RequirePackage{transparent}
\RequirePackage{caption}
\RequirePackage{emptypage} % hide footers on blank pages
\RequirePackage{calc} % TeX width calculations

\captionsetup{width=.75\textwidth,font=small,labelfont=bf}

% Use nicer PDF-based formatting
\usepackage{microtype}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MACROS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\def\@wbs{$<$n.m.o.p$>$}
\newcommand{\wbs}[1]{\def\@wbs{#1}}

\def\@projecttitle{Project Title}
\newcommand{\projecttitle}[1]{\def\@projecttitle{#1}}

\def\@milestoneid{$<$Milestone ID$>$}
\newcommand{\milestoneid}[1]{\def\@milestoneid{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HEADINGS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\geometry{
  margin=1in,
  headheight=1in}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\transparent{0.4}\includegraphics[height=0.4in]{ecp.eps}}
\fancyfoot[C]{\thepage}

\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FRONT MATTER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newif\if@mainmatter
\@mainmattertrue
\def\@clearifmainmatter{\if@mainmatter\relax\else\cleardoublepage\fi}

\newcommand\frontmatter{%
  \pagenumbering{roman}
  \@mainmatterfalse%
  %% --TITLE PAGE
  \begin{titlepage}
    \vspace*{1\baselineskip}%
    \begin{center}
      \includegraphics[width=5in]{ecp.eps}\\
      \vskip 3\baselineskip%
      \Large \textbf{Milestone Completion Report}\\
      \vskip 0.25\baselineskip%
      \textbf{WBS \@wbs}, \textbf{\@projecttitle}\\
      \vskip 1.00\baselineskip%
      \textbf{\@title}\\
      \vskip 0.25\baselineskip%
      \textbf{Milestone \@milestoneid}
      \vskip 1\baselineskip%
      \def\and{\\}% Newline between authors
      \setlength\tabcolsep{0pt}%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}%
      \vskip 1\baselineskip%
      \@date
    \end{center}
  \end{titlepage}%
  \newpage
%  \thispagestyle{empty}
} %% - end of FRONT MATTER

\renewenvironment{abstract}{%
  \@clearifmainmatter
  \addcontentsline{toc}{section}{Executive Summary}
  \titleformat{\section}{\bf\large}{\thetitle.}{1ex}{\MakeUppercase}[]
  \section*{Executive Summary}%
}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN MATTER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand\mainmatter{%
  \cleardoublepage
  \@mainmattertrue%
  \pagenumbering{arabic}}

\usepackage[bf,medium,raggedright]{titlesec}
\titleformat{\section}{\bf\large}{\thetitle.}{1ex}{\MakeUppercase}[]
\titleformat*{\subsection}{\bf}
\titleformat*{\subsubsection}{\bf}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPENDIX PAGE NUMBERING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\appendixpagenumbering}{
  \break
  % \pagenumbering{arabic}
  \renewcommand{\thepage}{\thesection-\arabic{page}}
}

% Change format for appendices: 'appendix' in title, title takes entire page,
% is not numbered, followed by a blank unnumbered page.
\renewcommand\appendixname{APPENDIX}
\renewcommand\appendix{%
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \newpage
  %
  %\titleclass{\section}{page}%
  %\assignpagestyle{\section}{empty}%
  %\titleformat{\section}[block]{\vfill\filcenter\bf}{\appendixname~\thetitle.}{1ex}{\MakeUppercase}[\vfill]%
  % \titlespacing*{\section}{0pt}{0pt}{0pt}%
  \gdef\thesection{\@Alph\c@section}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BIBLIOGRAPHY
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\refname{References}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end of report/ecpreport.cls
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
