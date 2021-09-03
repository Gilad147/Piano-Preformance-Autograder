\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % first lesson ex.2
% Thompson easiest piano course, part I, p. 13
% updated 3 May 2021

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \numericTimeSignature
  \time 4/4
  c1 | d1 | c2 d2 | c d | c4 d c d | c d c d | c c c c | c c c c \bar "|."}
    \new Staff = "lower" 
\relative c {
  \clef bass
  \numericTimeSignature
  \time 4/4
  R1 | R1 | R1 | R1 | R1 | R1 | R1 | R1 \bar "|." }
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
