import { Wheat } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Wheat size={16} />
          <span>© {new Date().getFullYear()} AgriSmart Advisor. All rights reserved.</span>
        </div>
        <div className="footer-links">
          <a href="#" onClick={(e) => e.preventDefault()}>Privacy Policy</a>
          <a href="#" onClick={(e) => e.preventDefault()}>Terms of Service</a>
        </div>
      </div>
    </footer>
  );
}
