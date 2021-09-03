\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % first lesson ex.7
% Thompson easiest piano course, part I, p. 21
% updated 3 May 2021
  
  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \numericTimeSignature
  \time 4/4
  	c4 d2 e4 | d2 c2 | d1 | d | 
	R1 | R1 | R1 | R1 | 
	c4 d2 e4 | d2 c2 | c4 d2 e4 | d2 c2 | 
	e4 d2 c4 | d2 e2 | c1 | R1 \bar "|."}
    \new Staff = "lower" 
\relative c {
  \clef bass
  \numericTimeSignature
  \time 4/4
   	R1 | R1 | R1 | R1 | 
	c'4 b2 a4 | b2 c2 | b1 | b |
	R1 | R1 | R1 | R1 |
	R1 | R1 | R1 | c1 \bar "|." }
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
