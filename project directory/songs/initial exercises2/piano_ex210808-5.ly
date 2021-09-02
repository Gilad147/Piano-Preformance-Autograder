\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % Alfred’s p. 21 Merrily we roll along

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

   <c e g>1 ~ <c e g>1 | <b f' g>1 | <c e g>1 | <c e g>1 ~ <c e g>1 | <b f' g>1 | <c e g>1  \bar "|."
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
