import React, { useEffect, useRef } from 'react';
import { Renderer, Stave, StaveNote, Formatter } from 'vexflow';

const MusicStaff = ({ notes }) => {
    const staffRef = useRef(null);

    useEffect(() => {
        const VF = require('vexflow');
        const renderer = new Renderer(staffRef.current, Renderer.Backends.SVG);
        renderer.resize(500, 200);
        const context = renderer.getContext();
        const stave = new Stave(10, 40, 400);
        stave.addClef("treble").setContext(context).draw();

        const staveNotes = notes.map(note => {
            return new StaveNote({
                keys: [note.key], // e.g., "c/4"
                duration: "q", // "q" for quarter note, "h" for half note, etc.
            });
        });

        Formatter.FormatAndDraw(context, stave, staveNotes);
    }, [notes]);

    return <div ref={staffRef} />;
};

export default MusicStaff;
