\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % first lesson ex.1
% Thompson easiest piano course, part I, p. 11
% updated 3 May 2021

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \time 4/4 \numericTimeSignature
  c4 c c c | R1 | c4 c c c | R1 \bar "|."}
    \new Staff = "lower" 
\relative c {
  \clef bass
  \time 4/4 \numericTimeSignature
  R1 | c'4 c c c | R1 | c4 c c c \bar "|." }
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
