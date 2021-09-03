\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % first lesson ex.10
% ascending and descending pentachord
% updated 8 July 2021

\new PianoStaff
<<
\new Staff = "upper"
\relative c' {
  \clef treble
  \numericTimeSignature
  \time 4/4
  	c1 | d1 | e1 | f1 | g1 | g1 | f1 | e1 | d1 | c1 \bar "|."}
\new Staff = "lower"
\relative c {
  \clef bass
  \numericTimeSignature
  \time 4/4
   	R1 | R1 | R1 | R1 | R1 | R1 | R1 | R1 | R1 | R1 \bar "|." }
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
