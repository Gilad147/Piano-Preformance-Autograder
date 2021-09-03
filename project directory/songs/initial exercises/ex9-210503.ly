\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % first lesson ex.9
% Based on Alfred's basic adult piano course, p. 11 (with modifications)
% updated 3 May 2021

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  	\clef treble
	\numericTimeSignature
  	\time 4/4
	c4 f e f | g d g2 | f4 e d e | f1 | c4 f e f | g d g2 | f4 e d e | f1 \bar "|."}
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
