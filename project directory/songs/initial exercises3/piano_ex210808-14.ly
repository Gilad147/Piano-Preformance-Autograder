\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % scale

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \key c \major
  \time 4/4

	%\override Score.BarLine.stencil = ##f
    c1 d e f g a b c 
    %\revert Score.BarLine.stencil
    \bar "|."
}
\new Staff = "lower" 

\relative c {
  \clef bass
  \key c \major
  \time 4/4

   %\override Score.BarLine.stencil = ##f
    c'1 b a g f e d c 
    %\revert Score.BarLine.stencil
    \bar "|." \bar "|."
}
>>

}
\layout {
  \context {
      \Score 
      proportionalNotationDuration =  #(ly:make-moment 1/5)
 }
 }
\midi {\tempo 4 = 60}
}
