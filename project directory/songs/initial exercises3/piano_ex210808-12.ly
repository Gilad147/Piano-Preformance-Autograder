\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % Alfred’s basic adult’s piano course, p. 21 

  \new PianoStaff 
  <<
    \new Staff = "upper" 

\relative c' {
  \clef treble
  \key c \major
  \time 4/4

	e4 d c d | e e e2 | d4 d d2 | e4 g g2 | e4 d c d | e e e2 | d4 d e d | c1 \bar "|."
}

\new Staff = "lower" 
\relative c {
  \clef bass
  \key c \major
  \time 4/4

   c1 | c1 | b1 | c1 | c1 | c1 | b1 | c1 \bar "|."
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
