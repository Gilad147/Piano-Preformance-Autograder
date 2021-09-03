\version "2.20.0"
\header {
\include "lilypond-book-preamble.ly"
   tagline = "" % removed
 }
\score {
 {
  % Alfred’s p. 19 Brother John

  \new PianoStaff 
  <<
    \new Staff = "upper" 
\relative c' {
  \clef treble
  \key c \major
  \time 4/4

	c4 d e c | c d e c | e f g2 | e4 f g2 | g4 f e c | g' f e c | c r4 c2 | c4 r c2 \bar "|."
}
\new Staff = "lower" 
\relative c {
  \clef bass
  \key c \major
  \time 4/4

   <c e g>1 | <c e g>1 | <c e g>1 | <c e g>1 | <c e g>1 | <c e g>1 | r4 g r2 | r4 g r2 \bar "|."
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
