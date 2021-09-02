\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % first lesson ex.4
% Thompson easiest piano course, part I, p. 16
% updated 3 May 2021

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \numericTimeSignature
  \time 2/4
  c4 d | R2 | c2 | R2 | c4 d | R2 | c2 | R2 \bar "|."}
    \new Staff = "lower" 
\relative c {
  \clef bass
  \numericTimeSignature
  \time 2/4
  R2 | c'4 b | R2 | c2 | R2 | c4 b | R2 | c2 \bar "|." }
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
